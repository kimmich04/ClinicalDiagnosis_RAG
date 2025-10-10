#!/usr/bin/env python3
"""
query_chroma.py

Query a Chroma collection using a saved vector from the embeddings folder or by encoding a text query
with the same CLIP model.

Usage examples:
  # query with saved vector (first vector of first *_text.npy found)
  python scripts/query_chroma.py --emb-dir Processed_embeddings/embeddings --collection documents --k 5

  # query with an inline text string (will load CLIP model to encode text)
  python scripts/query_chroma.py --collection documents --text "fever and rash" --k 5

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-dir', default='Processed_embeddings/embeddings')
    parser.add_argument('--collection', default='documents')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--sample', type=int, default=0, help='Index of vector inside sample file to query')
    parser.add_argument('--text', type=str, default=None, help='Optional text query to encode with CLIP')
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32', help='CLIP model to encode text queries')
    args = parser.parse_args()

    # choose query vector: either encode text or load an existing saved vector
    qvec = None
    if args.text:
        # encode using CLIP
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
        except Exception:
            raise SystemExit('Install transformers and torch to use text query: pip install transformers torch')
        model = CLIPModel.from_pretrained(args.model).to('cpu')
        processor = CLIPProcessor.from_pretrained(args.model)
        inputs = processor(text=[args.text], return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
        qvec = emb[0].cpu().numpy().tolist()
    else:
        emb_dir = Path(args.emb_dir)
        sample = next(emb_dir.glob('*_text.npy'), None)
        if sample is None:
            raise SystemExit('No text embedding files found in ' + str(emb_dir))
        arr = np.load(sample)
        if arr.size == 0:
            raise SystemExit('Sample file is empty: ' + str(sample))
        vec_idx = args.sample if args.sample < arr.shape[0] else 0
        qvec = arr[vec_idx].tolist()
        print(f'Querying with vector from {sample.name} index {vec_idx} (dim={len(qvec)})')

    # query chroma
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception:
        raise SystemExit('Install chromadb in your active environment: pip install chromadb')

    # try persistent duckdb+parquet store if possible
    try:
        client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory='chroma_db'))
    except Exception:
        client = chromadb.Client()

    # Diagnostics: list collections
    try:
        cols = client.list_collections()
        print('Chroma collections:', [c['name'] for c in cols])
    except Exception:
        try:
            # some clients return list of names
            print('Chroma collections:', client.list_collections())
        except Exception:
            pass

    try:
        collection = client.get_collection(args.collection)
    except Exception:
        collection = client.get_or_create_collection(args.collection)

    # Try to print collection size if available
    try:
        cnt = collection.count()
        # count() may return a dict or int depending on version
        if isinstance(cnt, dict):
            print('Collection counts:', cnt)
        else:
            print('Collection count:', cnt)
    except Exception:
        try:
            # fallback: try to get metadata length
            info = collection.count()
            print('Collection info:', info)
        except Exception:
            pass

    # perform the query
    print(f'Query vector length={len(qvec)}, first_values={qvec[:3]}')
    try:
        res = collection.query(query_embeddings=[qvec], n_results=args.k)
    except Exception as e:
        raise SystemExit('Chroma query failed: ' + str(e))

    print('Raw response type:', type(res))
    # Chroma may return a dict-like result
    if isinstance(res, dict):
        ids = res.get('ids', [[]])[0]
        dists = res.get('distances', [[]])[0]
        metas = res.get('metadatas', [[]])[0] if res.get('metadatas') else [None]*len(ids)
        docs = res.get('documents', [[]])[0] if res.get('documents') else [None]*len(ids)
        for i, id_row in enumerate(ids, 1):
            dist_row = dists[i-1] if i-1 < len(dists) else None
            meta = metas[i-1] if i-1 < len(metas) else None
            doc = docs[i-1] if i-1 < len(docs) else None
            print(f'{i}. id={id_row}, distance={dist_row}')
            if meta:
                print('   meta:', meta)
            if doc:
                print('   doc snippet:', str(doc)[:200])
    else:
        # Try object-like access
        try:
            ids = getattr(res, 'ids', None)
            distances = getattr(res, 'distances', None)
            metadatas = getattr(res, 'metadatas', None)
            documents = getattr(res, 'documents', None)
            if ids:
                for i, id_row in enumerate(ids[0], 1):
                    dist_row = distances[0][i-1] if distances and len(distances) > 0 else None
                    meta = metadatas[0][i-1] if metadatas and len(metadatas) > 0 else None
                    doc = documents[0][i-1] if documents and len(documents) > 0 else None
                    print(f'{i}. id={id_row}, distance={dist_row}')
                    if meta:
                        print('   meta:', meta)
                    if doc:
                        print('   doc snippet:', str(doc)[:200])
            else:
                print('Query returned (unrecognized format):', res)
        except Exception:
            print('Query returned (unable to pretty-print):', res)


if __name__ == '__main__':
    main()
