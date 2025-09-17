# src/phases/text_extraction/main.py
"""
Public entrypoint for Phase 1 text extraction.
Expose: run_text_extraction_pipeline(pdf_path) -> meta dict
"""

from .helpers.extractor import extract_text_and_meta
from .helpers.classifier import update_extract_and_clean

def run_text_extraction_pipeline(pdf_path: str) -> dict:
    """
    Orchestrates extraction and cleaning for a single PDF.
    Returns meta (including cleaned_text_updated and link_metadata).
    """
    meta = extract_text_and_meta(pdf_path)
    meta_updated = update_extract_and_clean(meta)
    return meta_updated
