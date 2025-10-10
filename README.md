Clinical-RAG — README

Overview
--------
This workspace extracts text chunks and images from clinical PDF case files, computes CLIP text and image embeddings, and ingests those vectors into a vector database for semantic retrieval.

Key directories
- `SourceMedicalRecords/` — original PDF files (source)
- `Processed/` — preprocessing output (markdown, images, metadata)
- `Processed_embeddings/embeddings/` — saved embeddings and JSON indexes (one JSON + two .npy files per document)
- `scripts/` — utilities and scripts (preprocessing, embedding, ingestion, query)

Important scripts
- `scripts/preprocessing_pdf.py` — (existing) extract pages/markdown/images/metadata from PDFs
- `scripts/embedding_clip.py` — compute CLIP text & image embeddings and save to `Processed_embeddings/embeddings`
- `scripts/check_embeddings.py` — validate saved `.npy` embedding files (shape, dtype, norms, cosines)
- `scripts/ingest_qdrant.py` — ingest embeddings into a Qdrant collection
- `scripts/query_qdrant.py` — example query script that searches the Qdrant collection using a saved vector (or a sample index)

Quick setup (recommended)
-------------------------
1) Create & activate a Python venv (macOS / Linux):

```bash
python -m venv venv
source venv/bin/activate
```

Windows (Command Prompt / PowerShell / WSL)
-----------------------------------------
Command Prompt:

```cmd
python -m venv venv
venv\Scripts\activate
```

PowerShell (if execution policy blocks activation, run PowerShell as Administrator and set-executionpolicy Unrestricted -Scope Process):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

WSL / Git Bash:

```bash
python -m venv venv
source venv/bin/activate
```

2) Install core dependencies (this repo uses a small set; add others as needed):

```bash
pip install -r requirements.txt
```

3) (Optional) If you plan to use Qdrant locally, start it with Docker:

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage -d qdrant/qdrant
```

Run the end-to-end workflow
---------------------------
1) Preprocess PDFs into markdown/images
- (if not already done) run the preprocessing script that produces `Processed/markdown` and `Processed/images`.

2) Compute embeddings

```bash
python scripts/embedding_clip.py --input-root Processed --output-root Processed_embeddings --model openai/clip-vit-base-patch32 --device cpu
```

Notes:
- Use `--device cuda` if you have GPU available and torch with CUDA.
- This script saves per-document files in `Processed_embeddings/embeddings/`:
  - `{stem}_text.npy` (text embeddings)
  - `{stem}_images.npy` (image embeddings)
  - `{stem}.json` (index listing chunks and image paths)

3) Validate embeddings (recommended)

```bash
python scripts/check_embeddings.py --emb-dir Processed_embeddings/embeddings --sample 5
```

This prints dtype, shape, norms, and pairwise cosines for each `.npy` file.

4) Ingest into Qdrant (example)

```bash
# start Qdrant (see above)
pip install qdrant-client
python scripts/ingest_qdrant.py --emb-dir Processed_embeddings/embeddings --collection documents
```

5) Run a sample query

```bash
python scripts/query_qdrant.py --emb-dir Processed_embeddings/embeddings --collection documents --k 5
```

This uses a vector taken from one saved file as the query vector and prints top-k hits with metadata.

Qdrant reports cosine similarity (by default) between your query vector and indexed vectors.
Cosine similarity ranges from:
1.0  → identical vectors (perfect match)
0.0  → orthogonal (no relation)
-1.0 → opposite meaning (rare in embeddings)