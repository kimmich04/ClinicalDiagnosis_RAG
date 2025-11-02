#!/usr/bin/env python3
"""
embedding_clip.py

Chunk markdowns under <input_root>/markdown and compute CLIP text & image embeddings.
Saves embeddings to <output_root>/embeddings.

python scripts/embedding_clip.py --input-root Processed --output-root Processed_embeddings --model openai/clip-vit-base-patch32 --device cpu

Usage:
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
    chunks = merge_short_chunks(chunks)
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

def merge_short_chunks(chunks, min_len=50):
    merged = []
    buffer = ""
    for c in chunks:
        if len(c["text"]) < min_len:
            buffer += " " + c["text"]
        else:
            if buffer:
                merged.append({"id": len(merged), "text": buffer.strip()})
                buffer = ""
            merged.append(c)
    if buffer:
        merged.append({"id": len(merged), "text": buffer.strip()})
    return merged


def clean_text(text: str) -> str:
    """Lightweight cleaning for OCR/extraction artifacts:
    - remove page header lines like 'page 12' or '# page 12'
    - collapse spaces between digits (e.g. '1 4' -> '14')
    - normalize spaces around hyphens (e.g. ' - ' -> '-')
    - collapse extra whitespace and trim
    """
    if not text:
        return text

    # Normalize unicode whitespace (non-breaking, zero-width) to regular spaces
    text = re.sub(r'[\u00A0\u200B\u202F]+', ' ', text)

    # Remove page header lines like '# Page 12' but avoid removing long lines
    # that accidentally begin with the word 'Page' followed by the article title
    # (some files have long single-line paragraphs). Filter line-by-line
    # and only drop short header lines.
    lines = text.splitlines()
    kept_lines = []
    for ln in lines:
        if re.match(r'(?i)^[#\s]*page\b', ln):
            # drop only if the line looks like a short page header
            if len(ln.strip()) < 120:
                continue
        kept_lines.append(ln)
    text = "\n".join(kept_lines)

    # Remove obvious image/file markers like '_2022_' or 'img 1'
    text = re.sub(r'[_\-]*\s*\d{4}\s*[_\-]*', ' ', text)
    text = re.sub(r'img\s*\d+', '', text, flags=re.IGNORECASE)

    # Remove underscores and multiple hyphens
    text = re.sub(r'[_\-]{2,}', '-', text)

    # Collapse spaced-out letters/numbers: '2 0 2 2' -> '2022', 'c a s e' -> 'case'
    text = re.sub(r'(?<=\b\w)\s+(?=\w\b)', '', text)

    # Remove isolated underscores and stray punctuation
    text = re.sub(r'[_/\\]+', ' ', text)

    # Normalize spaces around hyphens or commas
    text = re.sub(r'\s*-\s*', '-', text)
    text = re.sub(r'\s*,\s*', ', ', text)

    # Collapse multiple spaces or tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Remove leftover punctuation artifacts
    text = re.sub(r'[-–]{2,}', '-', text)
    text = re.sub(r'[()]', '', text)
    
    # Join split words from OCR (e.g. "infec- tion" → "infection")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Normalize spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # --- Additional safe fixes for numeric and OCR artifacts ---
    # Join digit groups that were split by spaces (e.g. '1 0 3' -> '103')
    # Join any whitespace between digits (covers NBSP/zero-width added above)
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    # Ensure a space before common medical units when they were glued to numbers
    text = re.sub(r'(?i)(\d+)(?=(mmhg|bpm|kg|cm|%))', r"\1 ", text)
    # Normalize common unit casing
    text = re.sub(r'(?i)\bmmhg\b', 'mmHg', text)
    # --- Blood-pressure reconstruction heuristic ---
    # Convert four-digit numbers followed by mmHg (e.g. '9060 mmHg') into '90/60 mmHg'
    # only when the split into two 2-digit numbers yields plausible BP values.
    def _format_bp(match):
        num = match.group(1)
        unit = match.group(2)
        # split into two 2-digit parts
        a = int(num[:2])
        b = int(num[2:])
        # sanity ranges: systolic 30-250, diastolic 20-200
        if 30 <= a <= 250 and 20 <= b <= 200:
            return f"{a}/{b} {unit}"
        return match.group(0)

    text = re.sub(r'\b(\d{4})\s*(mmHg)\b', _format_bp, text)
    # Fix decimals where spaces surround the dot: '39 . 6' -> '39.6'
    text = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text)
    # Remove spaces between digits and percent sign: '1 5 %' -> '15%'
    text = re.sub(r'(?<=\d)\s+%(?=\s|$)', '%', text)
    # Collapse spaced-out single letters/numbers left from OCR (e.g. 'c a s e' -> 'case')
    # But be conservative: only collapse when it's sequences of single letters separated by spaces
    text = re.sub(r'(?:(?<=\s)|^)(?:[A-Za-z0-9]\s+){2,}[A-Za-z0-9](?=(?:\s|$))',
                  lambda m: m.group(0).replace(' ', ''), text)

    # Remove stray multi-space clusters again
    text = re.sub(r'[ \t]{2,}', ' ', text)
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
    # out_emb.mkdir(parents=True, exist_ok=True)
    out_text = out_emb / 'text'; out_img = out_emb / 'image'
    out_text.mkdir(parents=True, exist_ok=True)
    out_img.mkdir(parents=True, exist_ok=True)

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

        # --- Remove image blocks / references BEFORE cleaning ---
        # Do this prior to clean_text so the cleaning rules don't accidentally
        # collapse or remove the main prose and leave only image markers.
        # Split at a heading like '## Images' (case-insensitive) if present and keep the part before it.
        parts = re.split(r"\n#{1,6}\s*images\b", text, flags=re.IGNORECASE)
        if parts:
            text = parts[0]

        # Remove inline markdown image references: ![alt](path)
        text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', ' ', text)
        # Remove stray image path fragments like '../images/...' or filenames ending in .jpeg/.jpg/.png
        text = re.sub(r'\.{2}/images/\S+', ' ', text)
        text = re.sub(r'\S+\.(?:jpe?g|png|webp)\b', ' ', text)

        # lightweight cleaning to fix OCR/extraction spacing artifacts
        text = clean_text(text)

        # Token-based chunking (preferred)
        chunks = tokenize_chunks(tokenizer, text, max_tokens=args.max_tokens, overlap=args.token_overlap)
        if not chunks:
            # fallback to char-chunking
            chunks = chunk_text_charwise(text, max_chars=args.max_chars, overlap=int(args.max_chars*0.1))

        texts = [c['text'] for c in chunks]
        # text embeddings saved under <output-root>/embeddings/text/
        text_emb_path = out_text / (stem + '_text.npy')

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
        # image embeddings saved under <output-root>/embeddings/image/
        image_emb_path = out_img / (stem + '_images.npy')
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
        # index = {
        #     'file': md.name,
        #     'n_chunks': len(chunks),
        #     'text_embedding': str(text_emb_path),
        #     'image_embedding': str(image_emb_path),
        #     'chunks': chunks,
        #     'images': image_list,
        # }
        index = {
            'file': md.name,
            'n_chunks': len(chunks),
            'text_embedding': str(text_emb_path),
            'image_embedding': str(image_emb_path),
            'chunks': [
                {'id': c['id'], 'text': c['text'], 'type': 'text'} for c in chunks
            ],
            'images': [{'path': p, 'type': 'image'} for p in image_list],
        }
        (out_emb / (stem + '.json')).write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append(index)
    metadata = {
        'model': args.model,
        'device': args.device,
        'num_files': len(summary),
        'embedding_dim_text': all_emb.shape[1] if len(summary) and 'text_embedding' in summary[-1] else 0,
        'embedding_dim_image': all_i_emb.shape[1] if len(summary) and 'image_embedding' in summary[-1] else 0
    }
    (out_emb / 'summary.json').write_text(json.dumps(metadata, indent=2))
    print('Saved embeddings to', out_emb)

if __name__ == '__main__':
    main()