#!/usr/bin/env python3
"""
check_embeddings.py

Quick validator for saved numpy embeddings produced by embedding_clip.py

Usage:
  python scripts/check_embeddings.py --emb-dir Processed_embeddings/embeddings --sample 5
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from math import isfinite


def cosine(a, b):
    if a.ndim != 1 or b.ndim != 1:
        return None
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def inspect_file(p: Path, sample_pairs: int = 5):
    try:
        arr = np.load(p)
    except Exception as e:
        print(f"ERROR loading {p}: {e}")
        return

    print(f"\nFile: {p.name}")
    print(f"  Path: {p}")
    print(f"  dtype: {arr.dtype}")
    print(f"  shape: {arr.shape}")
    if arr.size == 0:
        print("  -> empty array (0 embeddings)")
        return

    # numeric stats
    flat = arr.ravel()
    finite = np.all(np.isfinite(flat))
    print(f"  finite: {finite}")
    if not finite:
        print("  -> contains NaN/Inf values")

    mean = float(np.mean(flat))
    std = float(np.std(flat))
    print(f"  mean: {mean:.6e}, std: {std:.6e}")

    # check norms of first few vectors
    n_check = min(5, arr.shape[0])
    norms = [float(np.linalg.norm(arr[i])) for i in range(n_check)]
    print(f"  norms (first {n_check}): {[f'{v:.6e}' for v in norms]}")

    # cosine similarity among first few vectors (to detect identical vectors)
    n_pairs = min(sample_pairs, arr.shape[0])
    if arr.shape[0] >= 2:
        sims = []
        for i in range(n_pairs - 1):
            s = cosine(arr[i], arr[i+1])
            sims.append(None if s is None else f"{s:.6f}")
        print(f"  cosines (pairwise adjacent among first {n_pairs}): {sims}")

    # dimensionality check
    if arr.ndim != 2:
        print("  -> unexpected dimensionality (expected 2D array of embeddings)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-dir', default='Processed_embeddings/embeddings', help='Embeddings directory')
    parser.add_argument('--sample', type=int, default=5, help='Number of adjacent pairs to sample for cosine checks')
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    if not emb_dir.exists():
        print('Embeddings folder not found:', emb_dir)
        return

    files = sorted(list(emb_dir.glob('*.npy')))
    if not files:
        print('No .npy files found in', emb_dir)
        return

    print(f'Found {len(files)} .npy files in {emb_dir}')
    for p in files:
        inspect_file(p, sample_pairs=args.sample)

if __name__ == '__main__':
    main()
