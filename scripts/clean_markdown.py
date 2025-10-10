#!/usr/bin/env python3
"""
clean_markdown.py

Apply the project's clean_text normalization to all existing markdown files under Processed/markdown.
Creates a .bak copy for each file before overwriting.

Usage:
  python scripts/clean_markdown.py --md-dir Processed/markdown
"""
from __future__ import annotations
from pathlib import Path
import argparse
import re


def clean_text(text: str) -> str:
    if not text:
        return text
    text = text.replace('\u00A0', ' ')
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', text)
    text = re.sub(r'[–—―]+', '-', text)
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    text = re.sub(r'\s*-\s*', '-', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--md-dir', default='Processed/markdown')
    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    if not md_dir.exists():
        raise SystemExit('Markdown folder not found: ' + str(md_dir))

    files = sorted(md_dir.glob('*.md'))
    print(f'Found {len(files)} markdown files in {md_dir}')
    for p in files:
        src = p.read_text(encoding='utf-8')
        cleaned = clean_text(src)
        bak = p.with_suffix(p.suffix + '.bak')
        p.write_text(cleaned, encoding='utf-8')
        bak.write_text(src, encoding='utf-8')
        print(f'Cleaned {p.name} -> backup {bak.name}')

if __name__ == '__main__':
    main()
