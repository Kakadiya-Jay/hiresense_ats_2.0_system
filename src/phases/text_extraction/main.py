# src/phases/text_extraction/main.py
"""
Public entrypoint for Phase 1 text extraction.

This module previously exposed:
    run_text_extraction_pipeline(pdf_path: str) -> dict

The service layer expects:
    run_text_extraction_from_pdf_bytes(file_bytes: bytes) -> dict

To be compatible with both uses, we keep your existing pipeline function
and add a small wrapper that accepts raw PDF bytes, writes them to a temp file,
then calls the existing pipeline.

Make sure the rest of your helpers (helpers.extractor, helpers.classifier) are present.
"""

from typing import Dict, Any
import tempfile
import os

# import the existing pipeline function (keeps original behavior)
from .helpers.extractor import extract_text_and_meta
from .helpers.classifier import update_extract_and_clean


def run_text_extraction_pipeline(pdf_path: str) -> Dict[str, Any]:
    """
    Orchestrates extraction and cleaning for a single PDF file path.
    Returns meta (including cleaned_text_updated and link_metadata).
    This is your original function; kept here intact.
    """
    meta = extract_text_and_meta(pdf_path)
    meta_updated = update_extract_and_clean(meta)
    return meta_updated


# --- Compatibility wrapper expected by the service layer ---
def run_text_extraction_from_pdf_bytes(file_bytes: bytes) -> Dict[str, Any]:
    """
    Compatibility wrapper: accept PDF bytes, write to a temporary file,
    and call the existing run_text_extraction_pipeline(pdf_path) function.

    Returns the same dict as run_text_extraction_pipeline:
        {
            "cleaned_text_updated": "...",
            "link_metadata": [...],
            "emails": [...],
            "phones": [...]
        }
    """
    if not file_bytes:
        return {
            "cleaned_text_updated": "",
            "link_metadata": [],
            "emails": [],
            "phones": [],
        }

    # Create a temporary file with .pdf suffix that will be cleaned up
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()
        pdf_path = tmp.name

        # Call your existing pipeline that accepts a file path
        result = run_text_extraction_pipeline(pdf_path)

        # Ensure result is a dict and contains expected keys (best-effort normalization)
        if not isinstance(result, dict):
            # normalize to expected structure
            return {
                "cleaned_text_updated": str(result),
                "link_metadata": [],
                "emails": [],
                "phones": [],
            }

        # guarantee keys exist
        return {
            "cleaned_text_updated": result.get(
                "cleaned_text_updated", result.get("clean_text", "")
            ),
            "link_metadata": result.get("link_metadata", []),
            "emails": result.get("emails", []),
            "phones": result.get("phones", []),
            **{
                k: v
                for k, v in result.items()
                if k
                not in ["cleaned_text_updated", "link_metadata", "emails", "phones"]
            },
        }
    finally:
        # best-effort cleanup of temp file
        try:
            if tmp is not None:
                os.unlink(tmp.name)
        except Exception:
            pass
