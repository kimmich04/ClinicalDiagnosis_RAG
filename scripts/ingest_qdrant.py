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
  python scripts/ingest_qdrant.py --emb-dir Processed/embeddings --collection documents

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def ingest(emb_dir: Path, collection_name: str, host: str = 'localhost', port: int = 6333):
    client = QdrantClient(url=f'http://{host}:{port}')

    # find a sample text embedding to derive vector dimension
    sample = next(emb_dir.glob('*_text.npy'), None)
    if sample is None:
        raise SystemExit('No text embeddings found in ' + str(emb_dir))

    vec_dim = np.load(sample).shape[1]
    print(f'Using vector dimension = {vec_dim}')

    # create or recreate collection with cosine distance
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vec_dim, distance=rest.Distance.COSINE),
    )

    batch = []
    next_id = 1
    BATCH_SIZE = 256

    for jsonf in sorted(emb_dir.glob('*.json')):
        meta = json.loads(jsonf.read_text(encoding='utf-8'))
        stem = jsonf.stem
        text_npy = emb_dir / (stem + '_text.npy')
        img_npy = emb_dir / (stem + '_images.npy')

        # ingest text embeddings
        if text_npy.exists():
            arr = np.load(text_npy)
            for i, vec in enumerate(arr):
                payload = {
                    'file': meta.get('file'),
                    'type': 'text',
                    'chunk_id': i,
                    'text': meta.get('chunks', [])[i].get('text', '') if meta.get('chunks') else ''
                }
                batch.append(rest.PointStruct(id=next_id, vector=vec.tolist(), payload=payload))
                next_id += 1
                if len(batch) >= BATCH_SIZE:
                    client.upsert(collection_name=collection_name, points=batch)
                    batch = []

        # ingest image embeddings
        if img_npy.exists():
            arr = np.load(img_npy)
            for i, vec in enumerate(arr):
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

    if batch:
        client.upsert(collection_name=collection_name, points=batch)

    print(f'Ingest completed. Inserted points up to id {next_id - 1} into collection "{collection_name}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-dir', default='Processed_embeddings/embeddings', help='Embeddings directory')
    parser.add_argument('--collection', default='documents', help='Qdrant collection name')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    if not emb_dir.exists():
        raise SystemExit('Embeddings folder not found: ' + str(emb_dir))

    ingest(emb_dir, args.collection, args.host, args.port)
