import fitz  # PyMuPDF
import os
import json
import re

# -------- CONFIG --------
INPUT_DIR = "./SourceMedicalRecords"       # Folder containing your PDFs
OUTPUT_ROOT = "./Processed"          # Root output folder (will contain markdown/, images/, metadata/)
# ------------------------

MARKDOWN_DIR = os.path.join(OUTPUT_ROOT, "markdown")
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
METADATA_DIR = os.path.join(OUTPUT_ROOT, "metadata")

os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    if not text:
        return text

    # replace non-breaking spaces with normal spaces
    text = text.replace('\u00A0', ' ')

    # remove control characters (including those strange 0x01/0x03 artifacts)
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', text)

    # normalize various dash characters to simple hyphen
    text = re.sub(r'[–—―]+', '-', text)

    # collapse multiple whitespace into single space/newline preserving newlines
    # normalize newlines first
    text = re.sub(r'\r\n?', '\n', text)
    # collapse runs of spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # fix split numbers (like "4 8" -> "48")
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)

    # normalize hyphen spacing
    text = re.sub(r'\s*-\s*', '-', text)

    # collapse multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()


def process_pdf(pdf_path):
    """Extract text (plain text) + images from a PDF file."""
    doc = fitz.open(pdf_path)
    all_text = ""
    images = []

    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_image_dir = os.path.join(IMAGES_DIR, pdf_stem)
    os.makedirs(pdf_image_dir, exist_ok=True)

    for page_num, page in enumerate(doc, start=1):
        # ---- Extract text ----
        all_text += f"\n\n# Page {page_num}\n\n"
        # Use 'text' format which is widely supported across PyMuPDF versions
        all_text += page.get_text("text") or "[No text found]\n"

        # ---- Extract images ----
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image.get("image")
            image_ext = base_image.get("ext") or "png"
            image_name = f"{pdf_stem}_p{page_num}_img{img_index}.{image_ext}"
            image_path = os.path.join(pdf_image_dir, image_name)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            images.append({
                "page": page_num,
                "path": image_path
            })

    doc.close()
    return all_text.strip(), images


def main():
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ No PDFs found in {INPUT_DIR}")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        print(f"📄 Processing {pdf_file}...")

        text, images = process_pdf(pdf_path)

        # normalize extracted text to fix spacing/hyphenation artifacts
        md_text = clean_text(text)

        # Save Markdown file (one per PDF)
        md_filename = os.path.splitext(pdf_file)[0] + ".md"
        md_path = os.path.join(MARKDOWN_DIR, md_filename)
        # Simple heuristic: append image links at the end under an Images section
        if images:
            md_text += "\n\n## Images\n\n"
            for img in images:
                img_rel = os.path.relpath(img["path"], start=os.path.dirname(md_path))
                md_text += f"![figure]({img_rel})\n\n"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        # Save metadata JSON
        metadata = {
            "filename": pdf_file,
            "markdown_path": md_path,
            "images": images
        }

        json_path = os.path.join(METADATA_DIR, os.path.splitext(pdf_file)[0] + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Done → {os.path.relpath(md_path)} and {os.path.relpath(json_path)}")

    print("\n🎉 All PDFs processed successfully!")


if __name__ == "__main__":
    main()