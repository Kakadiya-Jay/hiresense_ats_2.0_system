# src/api/services/resume_service.py
"""
Resume service: orchestrates extraction (phase-1) and sectioning (phase-2).
Exposes `process_resume_payload` which accepts either:
 - extraction_json (already produced by phase-1) OR
 - pdf_bytes (raw PDF bytes) + filename
and returns the sectioned JSON.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# import text-extraction entrypoint: adjust if your project uses different path/name
try:
    from src.phases.text_extraction.main import run_text_extraction_from_pdf_bytes
except Exception:
    # fallback import path for older/alternate layout
    run_text_extraction_from_pdf_bytes = None

# import sectioning
from src.phases.sectioning.main import sectionaize_extracted_text


def process_resume_payload(
    extraction_json: Optional[Dict[str, Any]] = None,
    pdf_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    use_model_fallback: bool = True,
) -> Dict[str, Any]:
    """
    - If extraction_json provided: skip phase-1, run sectioning directly.
    - Else if pdf_bytes and run_text_extraction_from_pdf_bytes available: extract then sectionize.
    - Returns sectioned JSON (phase-2 wrapper).
    """
    if extraction_json is None and pdf_bytes is None:
        raise ValueError("Provide either extraction_json or pdf_bytes")

    # If extraction_json present: ensure expected keys, then sectionize
    if extraction_json is not None:
        logger.info("Processing provided extraction_json through sectioning")
        sectioned = sectionaize_extracted_text(
            extraction_json, use_model_fallback=use_model_fallback
        )
        # attach the original extraction (optionally)
        sectioned["raw_extraction"] = extraction_json
        return sectioned

    # Else try to extract from bytes
    if run_text_extraction_from_pdf_bytes is None:
        raise RuntimeError(
            "Text extraction function not found. Ensure src.text_extraction.main.run_text_extraction_from_pdf_bytes is implemented and importable."
        )

    # call extractor (it should return the phase-1 JSON)
    try:
        extraction_json = run_text_extraction_from_pdf_bytes(
            pdf_bytes,
        )
    except Exception as e:
        logger.exception("Text extraction failed: %s", e)
        raise

    sectioned = sectionaize_extracted_text(
        extraction_json, use_model_fallback=use_model_fallback
    )
    sectioned["raw_extraction"] = extraction_json
    return sectioned
