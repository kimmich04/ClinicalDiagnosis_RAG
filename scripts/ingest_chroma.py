#!/usr/bin/env python3
"""
ingest_chroma.py

Ingest embeddings from Processed_embeddings/embeddings into a local Chroma collection.

This script attempts to construct a Chroma client using the current API. If the
local Chroma installation requires migration, the script prints guidance.

Usage:
  pip install chromadb
  python scripts/ingest_chroma.py --emb-dir Processed_embeddings/embeddings --collection documents

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def make_client(persist_dir: str | None = None):
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:
        raise SystemExit("Please install chromadb in your active environment: pip install chromadb") from e

    # Try the modern explicit Settings construction first (works for many setups).
    if persist_dir:
        try:
            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
            client = chromadb.Client(settings)
            return client
        except ValueError as ve:
            # Older/newer versions of chroma may reject this config; fall back below
            print("Warning: chromadb.Settings rejected configuration (falling back to default client).")
            print(str(ve))

    # Fallback: try a default client instance
    try:
        client = chromadb.Client()
        return client
    except Exception as e:
        # If we get here, provide actionable instructions
        raise SystemExit(
            "Unable to create chromadb client. If you have an existing Chroma DB that needs migration,"
            " please run: pip install chroma-migrate && chroma-migrate\n"
            "Alternatively, upgrade/downgrade chromadb to a compatible version. See https://docs.trychroma.com/deployment/migration for details."
        ) from e


def ingest(emb_dir: Path, collection_name: str, persist_dir: str | None = None):
    client = make_client(persist_dir)

    try:
        # get or create collection
        collection = client.get_or_create_collection(name=collection_name)
    except Exception:
        # Some chroma versions use different names; try create_collection then get
        try:
            collection = client.create_collection(name=collection_name)
        except Exception as e:
            raise SystemExit("Failed to create or get Chroma collection: " + str(e))

    files = sorted(list(emb_dir.glob('*.json')))
    if not files:
        raise SystemExit('No index (.json) files found in ' + str(emb_dir))

    total = 0
    for jsonf in files:
        meta = json.loads(jsonf.read_text(encoding='utf-8'))
        stem = jsonf.stem
        text_npy = emb_dir / (stem + '_text.npy')
        img_npy = emb_dir / (stem + '_images.npy')

        # --- text embeddings ---
        if text_npy.exists():
            arr = np.load(text_npy)
            if arr.size:
                ids = [f"{stem}_text_{i}" for i in range(arr.shape[0])]
                docs = [c.get('text','') for c in meta.get('chunks', [])]
                metadatas = [
                    {
                        'file': meta.get('file'),
                        'chunk_id': c.get('id'),
                        'start_token': c.get('start_token'),
                        'end_token': c.get('end_token'),
                    }
                    for c in meta.get('chunks', [])
                ]
                embeddings = arr.astype(float).tolist()
                # Add to collection
                collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
                total += len(ids)

        # --- image embeddings ---
        if img_npy.exists():
            arr = np.load(img_npy)
            if arr.size:
                ids = [f"{stem}_img_{i}" for i in range(arr.shape[0])]
                docs = meta.get('images', [])
                metadatas = [{'file': meta.get('file'), 'image_path': p} for p in docs]
                embeddings = arr.astype(float).tolist()
                collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
                total += len(ids)

    # Persist if supported
    try:
        client.persist()
    except Exception:
        # Not all clients require/offer persist; ignore silently
        pass

    print(f'Ingest completed. Inserted {total} vectors into collection "{collection_name}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-dir', default='Processed_embeddings/embeddings', help='Embeddings directory')
    parser.add_argument('--collection', default='documents', help='Chroma collection name')
    parser.add_argument('--persist-dir', default='chroma_db', help='Chroma persist directory (duckdb+parquet)')
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    if not emb_dir.exists():
        raise SystemExit('Embeddings folder not found: ' + str(emb_dir))

    ingest(emb_dir, args.collection, args.persist_dir)
