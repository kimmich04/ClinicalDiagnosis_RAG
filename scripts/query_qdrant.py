#!/usr/bin/env python3
"""
query_qdrant.py

Query the Qdrant collection using a sample vector from the embeddings folder.

Usage:
  python scripts/query_qdrant.py --emb-dir Processed_embeddings/embeddings --collection documents --k 5
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-dir', default='Processed_embeddings/embeddings')
    parser.add_argument('--collection', default='documents')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--sample', type=int, default=0, help='Index of vector inside sample file to query')
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    # connect to local Qdrant server
    client = QdrantClient(url='http://localhost:6333')

    # pick a sample file: prefer embeddings/text/ subfolder, otherwise search recursively
    text_dir = emb_dir / 'text'
    if text_dir.exists():
        sample = next(text_dir.glob('*_text.npy'), None)
    else:
        sample = next(emb_dir.rglob('*_text.npy'), None)
    if sample is None:
        raise SystemExit('No text embedding files found in ' + str(emb_dir))

    arr = np.load(sample)
    if arr.size == 0:
        raise SystemExit('Sample file is empty: ' + str(sample))

    vec_idx = args.sample if args.sample < arr.shape[0] else 0
    qvec = arr[vec_idx].tolist()
    print(f'Querying with vector from {sample.name} index {vec_idx} (dim={len(qvec)})')

    # qdrant-client expects 'limit' (not 'top') and may return either objects or dicts
    results = client.search(collection_name=args.collection, query_vector=qvec, limit=args.k)
    print(f'Found {len(results)} hits:')
    for i, hit in enumerate(results, 1):
        # support both object-like and dict-like hits
        hid = getattr(hit, 'id', None) or (hit.get('id') if isinstance(hit, dict) else None)
        score = getattr(hit, 'score', None) or (hit.get('score') if isinstance(hit, dict) else None)
        payload = getattr(hit, 'payload', None) or (hit.get('payload') if isinstance(hit, dict) else None) or {}

        print(f'{i}. id={hid}, score={score}')
        if payload:
            file = payload.get('file')
            typ = payload.get('type')
            chunk = payload.get('chunk_id') if payload.get('chunk_id') is not None else payload.get('image_idx')
            txt = payload.get('text') if typ == 'text' else payload.get('image_path')
            print(f'   file={file}, type={typ}, idx={chunk}')
            if txt:
                def display_clean(s: str) -> str:
                    s = re.sub(r'(?mi)^\s*#?\s*page\b.*$', '', s)
                    s = re.sub(r'(?<=\d)\s+(?=\d)', '', s)
                    s = re.sub(r'(?<=\b[0-9A-Za-z])\s+(?=[0-9A-Za-z]\b)', '', s)
                    s = re.sub(r'\s*-\s*', '-', s)
                    s = re.sub(r'[ \t]+', ' ', s)
                    return s.strip()

                print(f"   snippet: {display_clean(str(txt))[:200]!r}")

if __name__ == '__main__':
    main()