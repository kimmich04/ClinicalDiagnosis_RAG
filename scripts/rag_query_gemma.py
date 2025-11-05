#!/usr/bin/env python3
"""
rag_query_gemma.py

Retrieve relevant embeddings from Qdrant and use Gemma (or any LLM) to generate an answer.
python scripts/rag_query_gemma.py
"""

from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np
import argparse
import json

# local helper for RAGAs evaluation (safe import)
# when run as a script the relative import can fail, so try absolute first
try:
    from ragas_eval import evaluate_with_ragas
except Exception:
    # fallback for package import
    from .ragas_eval import evaluate_with_ragas

# ---- CONFIG ----
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
TOP_K = 5  # number of relevant chunks to retrieve
EMBED_MODEL = "openai/clip-vit-base-patch32"  # for embeddings
GEMMA_MODEL = "google/medgemma-4b-it"  
# ----------------

# ---- EMBEDDING MODEL ----
def get_embedding_model():
    print("Loading CLIP embedding model...")
    model = CLIPModel.from_pretrained(EMBED_MODEL)
    processor = CLIPProcessor.from_pretrained(EMBED_MODEL)
    return model, processor

def get_text_embedding(model, processor, text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    # Normalize to unit vector
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy().tolist()

# ---- GEMMA MODEL ----
def get_gemma_pipeline():
    print("Loading Gemma model (this may take a while the first time)...")
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL, device_map="auto", torch_dtype="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# ---- QDRANT RETRIEVAL ----
def retrieve_context(qdrant_client, query_vec, top_k=TOP_K):
    print("Querying Qdrant for top results...")
    results = qdrant_client.query_points(  # modern method (replaces deprecated .search)
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k
    )
    context_chunks = []
    for point in results.points:
        payload = point.payload
        if payload.get("type") == "text" and payload.get("text"):
            context_chunks.append(payload["text"])
    return "\n\n".join(context_chunks)

# ---- MAIN LOOP ----
def main():
    parser = argparse.ArgumentParser(description="RAG query using Qdrant + Gemma; optional RAGAs evaluation")
    parser.add_argument("--eval", help="Path to evaluation JSONL/JSON file (list of {question, answers}) to run RAGAs evaluation")
    parser.add_argument("--example-index", type=int, default=None, help="If set, run evaluation only on this 0-based example index from the eval file")
    parser.add_argument("-k", "--top-k", dest="top_k", type=int, default=TOP_K, help="Number of retrieved chunks to use for evaluation or query")
    parser.add_argument("--mock", action="store_true", help="Run a fast mock evaluation that doesn't load models or query Qdrant (useful for smoke tests)")
    args = parser.parse_args()

    # If running mock evaluation, load data and run a lightweight evaluator without loading models
    if args.eval and args.mock:
        eval_path = args.eval
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
            with open(eval_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))

        # If user requested a single example, pick it and run only that one
        if args.example_index is not None:
            idx = args.example_index
            if not isinstance(idx, int) or idx < 0 or idx >= len(data):
                print(f"example-index {idx} out of range (0..{max(0, len(data)-1)})")
                return
            data = [data[idx]]

        # Mock evaluation: pretend retriever returns the gold_passages and generator returns the first gold answer
        print(f"Running MOCK evaluation on {len(data)} examples (top_k={args.top_k})")
        total_f1 = 0.0
        exact = 0
        n = 0
        def simple_f1(a, b):
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

        for ex in data:
            q = ex.get("question") or ex.get("query") or ex.get("prompt")
            golds = ex.get("answers") or ex.get("gold_answers") or []
            if not q or not golds:
                continue
            n += 1
            pred = golds[0]
            best_f1 = max(simple_f1(pred, g) for g in golds)
            total_f1 += best_f1
            if any(pred.strip().lower() == g.strip().lower() for g in golds):
                exact += 1

        avg_f1 = (total_f1 / n) if n else 0.0
        exact_pct = (exact / n * 100.0) if n else 0.0
        summary = {"examples": n, "avg_token_f1": avg_f1, "exact_match_pct": exact_pct}
        print("MOCK evaluation summary:", json.dumps(summary, indent=2))
        return

    # real run: create clients and load heavy models only when needed
    client = QdrantClient(url=QDRANT_URL)
    model, processor = get_embedding_model()
    gemma = get_gemma_pipeline()

    if args.eval:
        # load eval examples (support json or jsonl)
        eval_path = args.eval
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # try jsonl
            data = []
            with open(eval_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))

        # If user requested a single example, pick it and run only that one
        if args.example_index is not None:
            idx = args.example_index
            if not isinstance(idx, int) or idx < 0 or idx >= len(data):
                print(f"example-index {idx} out of range (0..{max(0, len(data)-1)})")
                return
            data = [data[idx]]

        def embed_fn(text: str):
            return get_text_embedding(model, processor, text)

        print(f"Running RAG evaluation on {len(data)} examples (top_k={args.top_k})")
        # Note: evaluate_with_ragas signature expects (qdrant_client, embed_fn, eval_examples, ...)
        # pass the eval examples as the third argument and provide the local generator as a fallback
        try:
            result = evaluate_with_ragas(client, embed_fn, data, collection_name=COLLECTION_NAME, top_k=args.top_k, generator_pipeline=gemma)
            # Print a concise summary of the evaluation result
            try:
                print("\nRAG evaluation result:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception:
                # If result is not JSON-serializable, just print its repr
                print(repr(result))
        except Exception as e:
            print(f"Evaluation failed: {e}")
        return

    # interactive mode
    while True:
        query = input("\n Ask a question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

            # Embed the query using CLIP text encoder
            query_vec = get_text_embedding(model, processor, query)
            context = retrieve_context(client, query_vec, top_k=args.top_k)

            if not context:
                print("⚠️ No relevant context found.")
                continue

            prompt = (
                f"Answer the following question using the given medical record context:\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\nAnswer:"
            )

            response = gemma(prompt, max_new_tokens=300, do_sample=False)
            print("\n💬 Answer:")
            print(response[0]["generated_text"].split("Answer:")[-1].strip())

if __name__ == "__main__":
    main()
