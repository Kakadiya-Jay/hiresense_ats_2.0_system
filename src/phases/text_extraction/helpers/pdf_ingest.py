# pdf_ingest.py
"""
PDF ingestion using PyMuPDF (fitz).
Reads a PDF and returns:
 - full_text: concatenated text
 - pages: list of per-page text
 - page_blocks: optional list of block dictionaries (page, bbox, text) for later layout-aware processing
"""
from typing import List, Dict, Tuple
import fitz  # pymupdf


def extract_pdf_text(path: str) -> Dict:
    doc = fitz.open(path)
    pages = []
    page_blocks = []
    for i, page in enumerate(doc):
        text = page.get_text("text")  # simple text extraction
        pages.append(text)
        # optional: capture block-level extraction (helpful for layout/columns)
        blocks = page.get_text("blocks")
        # Normalize blocks to dicts with bbox & text
        normalized_blocks = [
            {"page": i, "bbox": (b[0], b[1], b[2], b[3]), "text": b[4].strip()}
            for b in blocks
            if b[4].strip()
        ]
        page_blocks.append(normalized_blocks)
    full_text = "\n\n".join(pages)
    return {"full_text": full_text, "pages": pages, "page_blocks": page_blocks}


# quick demo:
if __name__ == "__main__":
    import sys

    res = extract_pdf_text(sys.argv[1])
    print("Pages:", len(res["pages"]))
    print(res["pages"][0][:1000])
