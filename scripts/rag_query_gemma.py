#!/usr/bin/env python3
"""
rag_query_gemma.py

Retrieve relevant embeddings from Qdrant and use Gemma (or any LLM) to generate an answer.
"""

from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np

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
def retrieve_context(qdrant_client, query_vec):
    print("Querying Qdrant for top results...")
    results = qdrant_client.query_points(  # modern method (replaces deprecated .search)
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=TOP_K
    )
    context_chunks = []
    for point in results.points:
        payload = point.payload
        if payload.get("type") == "text" and payload.get("text"):
            context_chunks.append(payload["text"])
    return "\n\n".join(context_chunks)

# ---- MAIN LOOP ----
def main():
    client = QdrantClient(url=QDRANT_URL)
    model, processor = get_embedding_model()
    gemma = get_gemma_pipeline()

    while True:
        query = input("\n Ask a question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        # Embed the query using CLIP text encoder
        query_vec = get_text_embedding(model, processor, query)
        context = retrieve_context(client, query_vec)

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
