#!/usr/bin/env python3
"""
ingest_qdrant.py

Ingest embeddings from Processed_embeddings/embeddings into a local Qdrant server.

Usage:
  # start Qdrant (one-time)
  docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage -d qdrant/qdrant

  # install client in venv
  pip install qdrant-client

  # run this script
  python scripts/ingest_qdrant.py --emb-dir Processed_embeddings/embeddings --collection documents

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Import helper from sibling module. When running this file as a script
# (python scripts/ingest_qdrant.py) the `scripts/` directory is on sys.path
# so import the module directly rather than as package `scripts.embedding_clip`.
from embedding_clip import clean_text


def ingest(emb_dir: Path, collection_name: str, host: str = 'localhost', port: int = 6333, dry_run: bool = False):
    """Ingest embeddings into Qdrant. If dry_run is True, do not connect to Qdrant and only report counts."""
    client = None
    if not dry_run:
        client = QdrantClient(url=f'http://{host}:{port}')

    # find a sample text embedding to derive vector dimension
    # prefer explicit text/ subfolder, otherwise search recursively
    text_dir = emb_dir / 'text'
    if text_dir.exists():
        sample = next(text_dir.glob('*_text.npy'), None)
    else:
        sample = next(emb_dir.rglob('*_text.npy'), None)
    if sample is None:
        raise SystemExit('No text embeddings found in ' + str(emb_dir))

    vec_dim = np.load(sample).shape[1]
    print(f'Using vector dimension = {vec_dim}')

    # create or recreate collection with cosine distance (skip when dry-run)
    if not dry_run:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vec_dim, distance=rest.Distance.COSINE),
        )

    batch = []
    next_id = 1
    BATCH_SIZE = 256

    total_candidates = 0
    total_text = 0
    total_images = 0

    for jsonf in sorted(emb_dir.glob('*.json')):
        meta = json.loads(jsonf.read_text(encoding='utf-8'))
        stem = jsonf.stem
        # text and image npy files are stored under text/ and image/ subfolders
        text_npy = emb_dir / 'text' / (stem + '_text.npy')
        img_npy = emb_dir / 'image' / (stem + '_images.npy')

        # ingest text embeddings (or count them in dry-run)
        if text_npy.exists():
            arr = np.load(text_npy)
            total_text += int(arr.shape[0])
            for i, vec in enumerate(arr):
                if dry_run:
                    # just count candidate
                    total_candidates += 1
                    continue
                payload = {
                    'file': meta.get('file'),
                    'type': 'text',
                    'chunk_id': i,
                    'text': clean_text(meta.get('chunks', [])[i].get('text', '') if meta.get('chunks') else '')
                }
                batch.append(rest.PointStruct(id=next_id, vector=vec.tolist(), payload=payload))
                next_id += 1
                if len(batch) >= BATCH_SIZE:
                    client.upsert(collection_name=collection_name, points=batch)
                    batch = []

        # ingest image embeddings (or count them in dry-run)
        if img_npy.exists():
            arr = np.load(img_npy)
            total_images += int(arr.shape[0])
            for i, vec in enumerate(arr):
                if dry_run:
                    total_candidates += 1
                    continue
                payload = {
                    'file': meta.get('file'),
                    'type': 'image',
                    'image_idx': i,
                    'image_path': meta.get('images', [])[i] if meta.get('images') else None
                }
                batch.append(rest.PointStruct(id=next_id, vector=vec.tolist(), payload=payload))
                next_id += 1
                if len(batch) >= BATCH_SIZE:
                    client.upsert(collection_name=collection_name, points=batch)
                    batch = []

    if not dry_run and batch:
        client.upsert(collection_name=collection_name, points=batch)

    if dry_run:
        print('Dry-run completed. No data was sent to Qdrant.')
        print(f'  Files inspected: {len(list(emb_dir.glob("*.json")))}')
        print(f'  text vectors total: {total_text}')
        print(f'  image vectors total: {total_images}')
        print(f'  total candidate points: {total_candidates}')
    else:
        print(f'Ingest completed. Inserted points up to id {next_id - 1} into collection "{collection_name}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-dir', default='Processed_embeddings/embeddings', help='Embeddings directory')
    parser.add_argument('--collection', default='documents', help='Qdrant collection name')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--dry-run', action='store_true', help='Do not connect to Qdrant; just validate files and print counts')
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    if not emb_dir.exists():
        raise SystemExit('Embeddings folder not found: ' + str(emb_dir))

    ingest(emb_dir, args.collection, args.host, args.port, dry_run=args.dry_run)