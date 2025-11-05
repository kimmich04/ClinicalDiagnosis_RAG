"""ragas_eval.py

Minimal RAGAs evaluation helper that uses Google Gemini as the LLM
and a provided embedding function for RAG metric computation.

Requirements (runtime):
- ragas installed in the same environment
- google-generative-ai (pip package name: google-generative-ai)
- set environment variable GOOGLE_API_KEY to a valid Gemini API key

This module implements a single top-level function:
- evaluate_with_ragas(qdrant_client, embed_fn, gemini_model, eval_examples, ...)

The function will:
- generate "response" for any example missing it using Gemini
- convert examples into a RAGAs EvaluationDataset
- build a ragas-compatible LLM via ragas.llms.llm_factory(provider='google') using
  the google.generativeai client
- build a thin embeddings adapter around the provided embed_fn
- call ragas.evaluate(dataset, llm=..., embeddings=...)

The implementation is defensive and prints helpful messages when dependencies
or credentials are missing.
"""
from typing import List, Dict, Any, Optional
import os
import json

def _ensure_genai_configured():
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise ImportError(
            "google.generativeai is required for Gemini-backed evaluation: pip install google-generative-ai"
        ) from e

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable must be set to use Gemini")

    try:
        # configure if function exists
        if hasattr(genai, "configure"):
            genai.configure(api_key=api_key)
    except Exception:
        # ignore configuration errors here; genai may also support other auth flows
        pass
    return genai


class RagasEmbeddingsAdapter:
    """Thin adapter exposing embed_query and embed_documents for ragas metrics.

    embed_fn should be a callable text->vector (list[float] or numpy array).
    """
    def __init__(self, embed_fn):
        self._embed = embed_fn

    def embed_query(self, text: str):
        return self._embed(text)

    def embed_documents(self, docs: List[str]):
        return [self._embed(d) for d in docs]


def _generate_with_gemini(genai, model: str, prompt: str) -> str:
    # The google.generativeai package has changed APIs across versions. Try a few
    # common call patterns and extract text robustly.
    last_err = None

    # 1) genai.chat.completions.create(...) (newer API shape)
    try:
        if hasattr(genai, "chat") and hasattr(genai.chat, "completions") and hasattr(genai.chat.completions, "create"):
            resp = genai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            # try common fields
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                return getattr(cand, "content", str(cand))
            if hasattr(resp, "choices") and resp.choices:
                ch = resp.choices[0]
                # choice may contain a message or text
                if hasattr(ch, "message") and getattr(ch.message, "content", None):
                    return ch.message.content
                return getattr(ch, "text", str(ch))
    except Exception as e:
        last_err = e

    # 2) genai.generate(...) (some releases)
    try:
        if hasattr(genai, "generate"):
            resp = genai.generate(model=model, prompt=prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text
            if hasattr(resp, "candidates") and resp.candidates:
                c = resp.candidates[0]
                return getattr(c, "content", str(c))
            return str(resp)
    except Exception as e:
        last_err = e

    # 3) genai.client.generate_text(...) or genai.generate_text(...) (older/alternate)
    try:
        client = getattr(genai, "client", None)
        if client and hasattr(client, "generate_text"):
            out = client.generate_text(model=model, input=prompt)
            # many shapes: out.output[0].content or out.text
            if hasattr(out, "output") and out.output:
                return getattr(out.output[0], "content", str(out.output[0]))
            if hasattr(out, "text") and out.text:
                return out.text
            return str(out)
        if hasattr(genai, "generate_text"):
            out = genai.generate_text(model=model, input=prompt)
            if hasattr(out, "output") and out.output:
                return getattr(out.output[0], "content", str(out.output[0]))
            if hasattr(out, "text") and out.text:
                return out.text
            return str(out)
    except Exception as e:
        last_err = e

    # 4) fallback: try chat completions via a .create on genai.chat.completions with different args
    try:
        if hasattr(genai, "chat") and hasattr(genai.chat, "completions"):
            resp = genai.chat.completions.create(model=model, messages=[{"content": prompt}])
            if hasattr(resp, "candidates") and resp.candidates:
                return getattr(resp.candidates[0], "content", str(resp.candidates[0]))
    except Exception as e:
        last_err = e

    # If we reach here, we couldn't generate text
    # Provide a short description of the genai object to aid debugging
    def _describe_genai(g):
        names = []
        try:
            names = sorted([n for n in dir(g) if not n.startswith("__")])
        except Exception:
            names = []
        flags = {
            "has_chat": hasattr(g, "chat"),
            "has_generate": hasattr(g, "generate"),
            "has_generate_text": hasattr(g, "generate_text"),
            "has_client": hasattr(g, "client"),
        }
        return f"flags={flags}, attrs={names[:30]}"

    desc = _describe_genai(genai)
    raise RuntimeError(
        "Gemini generation failed; last error: "
        + (str(last_err) if last_err is not None else "unknown")
        + "; genai_info="
        + desc
    )


def evaluate_with_ragas(
    qdrant_client,
    embed_fn,
    eval_examples: List[Dict[str, Any]],
    collection_name: str = "documents",
    top_k: int = 5,
    gemini_model: Optional[str] = None,
    generator_pipeline=None,
    verbose: bool = True,
):
    """Run RAG evaluation using RAGAs with Gemini as the LLM and the provided embeddings.

    Args:
        qdrant_client: a QdrantClient instance (used only to fetch context for generation)
        embed_fn: callable(text) -> vector. Used to build embeddings adapter for ragas and to retrieve contexts
        eval_examples: list of dicts. Each dict should contain at least 'question' and 'answers' (list[str]).
        collection_name: Qdrant collection name
        top_k: number of retrieved chunks to attach when generating responses
        gemini_model: model name to pass to Gemini (defaults to env GEMINI_MODEL or 'gemini-pro')
        verbose: print progress and diagnostics when True

    Returns:
        result from ragas.evaluate(...) or raises an informative error.
    """
    # import ragas lazily to give clear errors
    try:
        import ragas  # type: ignore
    except Exception as e:
        raise ImportError("ragas package is required for this evaluation. Install it into the environment.") from e

    genai = _ensure_genai_configured()
    model = gemini_model or os.environ.get("GEMINI_MODEL", "gemini-pro")

    # Some versions of google.generativeai expose a convenience function
    # `get_model(name)` which returns a model object. The instructor/integration
    # used by ragas expects an instance of genai.GenerativeModel (or a patched
    # equivalent). Try several strategies to get a correct instance:
    # 1. Prefer genai.get_model(model) if it returns a GenerativeModel-like object.
    # 2. Otherwise try to instantiate genai.GenerativeModel(model_name=...)
    # 3. If we obtained a candidate model object but it's of a different
    #    type, try to patch it with `instructor.patch(...)` (if instructor is
    #    installed) to obtain the expected shape.
    genai_client = None
    candidate = None

    # 1) Try genai.get_model(...) if available
    if hasattr(genai, "get_model"):
        try:
            candidate = genai.get_model(model)
            genai_client = candidate
        except Exception:
            candidate = None

    # 2) If no client yet, try constructing GenerativeModel(model_name=...)
    if genai_client is None and hasattr(genai, "GenerativeModel"):
        try:
            # Prefer the `model_name` kwarg which some genai versions expect
            genai_client = genai.GenerativeModel(model_name=model)
        except Exception:
            try:
                # Fallback to `model` kwarg for other versions
                genai_client = genai.GenerativeModel(model=model)
            except Exception:
                genai_client = None

    # 3) If we have a candidate that's not a GenerativeModel, try patching
    # it with instructor.patch(...) so it matches the expected interface.
    if genai_client is None and candidate is not None:
        try:
            import instructor  # type: ignore
            if hasattr(instructor, "patch"):
                try:
                    patched = instructor.patch(candidate)
                    genai_client = patched
                except Exception:
                    # instructor.patch may raise if candidate incompatible
                    genai_client = None
        except Exception:
            # instructor not available or patch failed
            genai_client = None

    # If still no client, raise a clear error with guidance
    if genai_client is None:
        raise RuntimeError(
            "Could not construct a google.generativeai GenerativeModel instance or a patched equivalent. "
            "Ensure you have a recent `google-generative-ai` package that exposes `get_model(model_name)` or `GenerativeModel`, "
            "and install `instructor` if required to patch model objects."
        )

    # Build retrieval helper (query Qdrant for top-k passages)
    class _Retriever:
        def __init__(self, client, collection):
            self.client = client
            self.collection = collection

        def retrieve(self, query_vec, k):
            results = self.client.query_points(collection_name=self.collection, query=query_vec, limit=k)
            chunks = []
            for p in results.points:
                payload = p.payload
                if payload and payload.get("type") == "text" and payload.get("text"):
                    chunks.append(payload["text"])
            return chunks

    retriever = _Retriever(qdrant_client, collection_name)

    # Ensure each sample has a 'response' by generating via Gemini (we use retrieval to build context)
    for ex in eval_examples:
        if ex.get("response"):
            continue
        q = ex.get("question") or ex.get("query") or ex.get("prompt")
        if not q:
            continue
        try:
            q_vec = embed_fn(q)
        except Exception:
            # embed_fn may return numpy array; convert when necessary
            q_vec = embed_fn(q)
        chunks = retriever.retrieve(q_vec, top_k)
        context = "\n\n".join(chunks)
        if verbose:
            print(f"Retrieved {len(chunks)} chunks for generation. Context preview:\n{context[:400]}")
        prompt = f"Answer the following question using the given context:\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:"
        if verbose:
            print(f"Generating response for question: {q[:80]}")
        # If the installed google.generativeai client doesn't expose a generation
        # API (some SDK builds differ), fall back to a provided local generator
        # (e.g., HF pipeline) if available.
        try:
            resp_text = _generate_with_gemini(genai, model, prompt)
        except RuntimeError as e:
            # If we have a local generator pipeline, use it as a fallback.
            if generator_pipeline is not None:
                try:
                    out = generator_pipeline(prompt, max_new_tokens=256, do_sample=False)
                    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "generated_text" in out[0]:
                        resp_text = out[0]["generated_text"]
                    else:
                        resp_text = str(out)
                except Exception as e2:
                    raise RuntimeError(f"Gemini generation failed ({e}); local generator fallback also failed: {e2}") from e2
            else:
                raise RuntimeError(
                    f"Gemini generation failed and no local generator provided. Original error: {e}"
                ) from e
        ex["response"] = resp_text

    # Convert to RAGAs EvaluationDataset
    try:
        EvaluationDataset = ragas.EvaluationDataset
    except Exception:
        from ragas.dataset_schema import EvaluationDataset  # type: ignore

    ragas_list = []
    for ex in eval_examples:
        q = ex.get("question") or ex.get("query") or ex.get("prompt")
        if not q:
            continue
        sample = {"user_input": q}
        if ex.get("response") is not None:
            sample["response"] = ex.get("response")
        golds = ex.get("answers") or ex.get("gold_answers") or []
        if isinstance(golds, list) and golds:
            sample["reference"] = golds[0]
        contexts = ex.get("gold_passages") or ex.get("reference_contexts") or []
        if contexts:
            sample["reference_contexts"] = contexts
        if ex.get("id"):
            sample["id"] = ex.get("id")
        ragas_list.append(sample)

    try:
        dataset_obj = EvaluationDataset.from_list(ragas_list)
    except Exception as e:
        raise RuntimeError(f"Failed to build RAGAs EvaluationDataset: {e}")

    # Create ragas-compatible LLM via llm_factory(provider='google') using genai client
    try:
        from ragas.llms import llm_factory  # type: ignore
        llm_obj = llm_factory(model=model, provider="google", client=genai_client)
    except Exception as e:
        # If we cannot build a ragas-compatible Gemini LLM (SDK differences),
        # fall back to a simple local evaluator that compares the generated
        # responses (we already generated them above) to the references using
        # token-level F1 and exact match. This ensures the user still gets
        # useful evaluation results without requiring a perfect instructor/genai
        # object shape.
        if verbose:
            print(
                "Warning: could not construct a RAGAs Gemini LLM (will run local fallback metrics)."
            )
            print(f"Original llm_factory error: {e}")

        def _simple_f1(a: str, b: str) -> float:
            atoks = [t for t in a.lower().split() if t]
            btoks = [t for t in b.lower().split() if t]
            if not atoks or not btoks:
                return 0.0
            common = 0
            bcounts = {}
            for t in btoks:
                bcounts[t] = bcounts.get(t, 0) + 1
            for t in atoks:
                if bcounts.get(t, 0) > 0:
                    common += 1
                    bcounts[t] -= 1
            if common == 0:
                return 0.0
            p = common / len(atoks)
            r = common / len(btoks)
            return 2 * p * r / (p + r)

        total_f1 = 0.0
        exact = 0
        n = 0
        for ex in eval_examples:
            pred = ex.get("response") or ""
            golds = ex.get("answers") or ex.get("gold_answers") or []
            if not golds:
                continue
            n += 1
            best_f1 = max(_simple_f1(pred, g) for g in golds)
            total_f1 += best_f1
            if any(pred.strip().lower() == g.strip().lower() for g in golds):
                exact += 1

        avg_f1 = (total_f1 / n) if n else 0.0
        exact_pct = (exact / n * 100.0) if n else 0.0
        result = {"examples": n, "avg_token_f1": avg_f1, "exact_match_pct": exact_pct, "note": "fallback_local_metrics"}
        return result

    emb_adapter = RagasEmbeddingsAdapter(embed_fn)

    # Call ragas.evaluate with explicit llm and embeddings to avoid automatic defaults
    if verbose:
        print("Calling ragas.evaluate(...) with Gemini LLM and provided embeddings...")
    try:
        result = ragas.evaluate(dataset_obj, llm=llm_obj, embeddings=emb_adapter)
        if verbose:
            print("RAGAs evaluation completed. Returning result.")
        return result
    except Exception as e:
        # Provide a helpful error message
        raise RuntimeError(f"ragas.evaluate(...) failed: {e}")
