#!/usr/bin/env python3
"""
embedding_clip.py

Chunk markdowns under <input_root>/markdown and compute CLIP text & image embeddings.
Saves embeddings to <output_root>/embeddings.

Usage:
  python embedding_clip.py --input-root Processed --output-root Processed --model openai/clip-vit-base-patch32 --device cpu
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re

def tokenize_chunks(tokenizer, text: str, max_tokens:int, overlap:int):
    # token_ids = tokenizer.encode(text, add_special_tokens=False)
    encoded = tokenizer(text, add_special_tokens=False)['input_ids']
    chunks = []
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    i = 0
    cid = 0
    while i < len(encoded):
        chunk_ids = encoded[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append({'id': cid, 'text': chunk_text, 'start_token': i, 'end_token': i+len(chunk_ids)})
        cid += 1
        if i + max_tokens >= len(encoded):
            break
        i += step
    return chunks

def chunk_text_charwise(text: str, max_chars:int, overlap:int):
    if not text:
        return []
    chunks = []
    start = 0
    step = max_chars - overlap if max_chars > overlap else max_chars
    cid = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        snippet = text[start:end].strip()
        if snippet:
            chunks.append({'id': cid, 'text': snippet, 'start_char': start, 'end_char': end})
            cid += 1
        if end == len(text):
            break
        start += step
    return chunks


def clean_text(text: str) -> str:
    """Lightweight cleaning for OCR/extraction artifacts:
    - remove page header lines like 'page 12' or '# page 12'
    - collapse spaces between digits (e.g. '1 4' -> '14')
    - normalize spaces around hyphens (e.g. ' - ' -> '-')
    - collapse extra whitespace and trim
    """
    if not text:
        return text

    # remove page headers (lines that start with 'page' or '# page')
    text = re.sub(r'(?mi)^[#\s]*page\b.*$', '', text)

    # collapse spaces between digits: '1 4' -> '14'
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)

    # collapse spaces between single alnum characters: '4 8 a 3 1' -> '48a31'
    # match a space where both neighbors are single alnum characters (word-boundary)
    text = re.sub(r'(?<=\b[0-9A-Za-z])\s+(?=[0-9A-Za-z]\b)', '', text)

    # normalize hyphens used as connectors: ' - ' or ' -' -> '-'
    text = re.sub(r'\s*-\s*', '-', text)

    # collapse multiple spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # collapse multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', default='Processed', help='Root with markdown/ and images/')
    parser.add_argument('--output-root', default='Processed', help='Root to write embeddings into <output-root>/embeddings/')
    parser.add_argument('--model', default='openai/clip-vit-base-patch32', help='HF CLIP model name')
    parser.add_argument('--device', default='cpu', help='torch device (cpu or cuda)')
    parser.add_argument('--max-tokens', type=int, default=77, help='Max tokens per text chunk (CLIP default 77)')
    parser.add_argument('--token-overlap', type=int, default=10, help='Token overlap between chunks')
    parser.add_argument('--max-chars', type=int, default=1000, help='(fallback) char chunk size when tokenizer unavailable')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    markdown_dir = input_root / 'markdown'
    images_root = input_root / 'images'
    out_emb = Path(args.output_root) / 'embeddings'
    out_emb.mkdir(parents=True, exist_ok=True)

    if not markdown_dir.exists():
        print('Markdown folder not found:', markdown_dir)
        return

    # Load CLIP via transformers
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch
    except Exception as e:
        raise RuntimeError('Install transformers and torch: pip install transformers torch') from e

    model = CLIPModel.from_pretrained(args.model).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer  # use tokenizer for token-based chunking

    summary = []
    for md in sorted(markdown_dir.glob('*.md')):
        stem = md.stem
        text = md.read_text(encoding='utf-8')
        # lightweight cleaning to fix OCR/extraction spacing artifacts
        text = clean_text(text)

        # Token-based chunking (preferred)
        chunks = tokenize_chunks(tokenizer, text, max_tokens=args.max_tokens, overlap=args.token_overlap)
        if not chunks:
            # fallback to char-chunking
            chunks = chunk_text_charwise(text, max_chars=args.max_chars, overlap=int(args.max_chars*0.1))

        texts = [c['text'] for c in chunks]
        text_emb_path = out_emb / (stem + '_text.npy')

        if texts:
            # process in batches to avoid OOM
            batch_size = 16
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = processor(text=batch, return_tensors='pt', padding=True, truncation=True).to(args.device)
                with torch.no_grad():
                    emb = model.get_text_features(**inputs)
                all_embs.append(emb.cpu().numpy())
            all_emb = np.vstack(all_embs)
        else:
            # model.text_projection may be a torch.nn.Linear (has out_features) or a Tensor.
            try:
                out_dim = int(model.text_projection.out_features)
            except Exception:
                try:
                    out_dim = int(model.text_projection.weight.shape[0])
                except Exception:
                    # fallback: try numpy shape attribute (if it's a Tensor)
                    out_dim = int(getattr(model.text_projection, 'shape', (0, 0))[1])
            all_emb = np.zeros((0, out_dim), dtype=np.float32)

        np.save(text_emb_path, all_emb)

        # images
        img_dir = images_root / stem
        image_emb_path = out_emb / (stem + '_images.npy')
        image_list = []
        if img_dir.exists():
            image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')])
            if image_paths:
                batch_size = 8
                all_i_emb = []
                from PIL import Image
                for i in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[i:i+batch_size]
                    images = [Image.open(p).convert('RGB') for p in batch_paths]
                    inputs = processor(images=images, return_tensors='pt').to(args.device)
                    with torch.no_grad():
                        i_emb = model.get_image_features(**inputs)
                    all_i_emb.append(i_emb.cpu().numpy())
                all_i_emb = np.vstack(all_i_emb)
                np.save(image_emb_path, all_i_emb)
                image_list = [str(p) for p in image_paths]
            else:
                try:
                    out_dim = int(model.visual_projection.out_features)
                except Exception:
                    try:
                        out_dim = int(model.visual_projection.weight.shape[0])
                    except Exception:
                        out_dim = int(getattr(model.visual_projection, 'shape', (0, 0))[1])
                np.save(image_emb_path, np.zeros((0, out_dim), dtype=np.float32))
        else:
            try:
                out_dim = int(model.visual_projection.out_features)
            except Exception:
                try:
                    out_dim = int(model.visual_projection.weight.shape[0])
                except Exception:
                    out_dim = int(getattr(model.visual_projection, 'shape', (0, 0))[1])
            np.save(image_emb_path, np.zeros((0, out_dim), dtype=np.float32))

        # Write index for this file
        index = {
            'file': md.name,
            'n_chunks': len(chunks),
            'text_embedding': str(text_emb_path),
            'image_embedding': str(image_emb_path),
            'chunks': chunks,
            'images': image_list,
        }
        (out_emb / (stem + '.json')).write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append(index)

    (out_emb / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved embeddings to', out_emb)

if __name__ == '__main__':
    main()