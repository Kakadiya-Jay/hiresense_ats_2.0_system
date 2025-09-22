# sec/api/services/resume_service.py
"""
Resume service: orchestrates extraction (phase-1), sectioning (phase-2),
and optional feature-extraction (phase-3).

Public function:
 - process_resume_payload(...)
    - If extraction_json provided: skip text extraction and run sectioning.
    - Else if pdf_bytes provided: run text extraction (if available) then sectioning.
    - Optionally runs feature-extraction (if the module is importable).
    - Returns dict with keys: "sectioned" and, when requested and available, "features".
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# phase-1 text extraction entrypoint (optional)
try:
    from src.phases.text_extraction.main import run_text_extraction_from_pdf_bytes
except Exception:
    run_text_extraction_from_pdf_bytes = None
    logger.debug(
        "run_text_extraction_from_pdf_bytes not importable; assuming extraction_json will be provided."
    )

# phase-2 sectioning (required)
try:
    from src.phases.sectioning.main import sectionaize_extracted_text
except Exception as e:
    sectionaize_extracted_text = None
    logger.exception("sectionaize_extracted_text import failed: %s", e)

# phase-3 feature extraction (optional)
try:
    from src.phases.feature_extraction.main import extract_features_from_sectioned
except Exception:
    extract_features_from_sectioned = None
    logger.debug("Feature extraction module not available; continuing without phase-3.")


def process_resume_payload(
    extraction_json: Optional[Dict[str, Any]] = None,
    pdf_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    use_model_fallback: bool = True,
    run_feature_extraction: bool = True,
    jd_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrate resume processing.

    Returns a dict containing at minimum:
      {
        "status": "ok",
        "sectioned": { ... }            # phase-2 output (if produced / provided)
        "features": { ... }             # phase-3 output (if requested and available)
      }

    Args:
      extraction_json: if provided, should be phase-1 output and will be sectioned.
      pdf_bytes: raw PDF bytes; used when extraction_json not provided.
      filename: optional filename for logging / heuristics.
      use_model_fallback: passed to sectioning.
      run_feature_extraction: when True and feature-extraction available, run it.
      jd_text: optional job-description text to pass to feature extractor for semantic matching.
    """

    if extraction_json is None and pdf_bytes is None:
        raise ValueError("Provide either extraction_json or pdf_bytes")

    # Ensure sectioning function exists
    if sectionaize_extracted_text is None:
        raise RuntimeError(
            "Sectioning function not found. Ensure src.phases.sectioning.main.sectionaize_extracted_text is implemented and importable."
        )

    # If extraction_json provided: skip text extraction
    if extraction_json is not None:
        logger.info("Processing provided extraction_json through sectioning")
        sectioned = sectionaize_extracted_text(
            extraction_json, use_model_fallback=use_model_fallback
        )
        # attach original extraction (useful for debugging)
        sectioned["raw_extraction"] = extraction_json
    else:
        # pdf_bytes path: ensure extractor exists
        if run_text_extraction_from_pdf_bytes is None:
            raise RuntimeError(
                "Text extraction function not found. Provide extraction_json or implement src.phases.text_extraction.main.run_text_extraction_from_pdf_bytes"
            )
        try:
            logger.info("Running text extraction on provided PDF bytes")
            extraction_json = run_text_extraction_from_pdf_bytes(
                pdf_bytes,
            )
        except Exception as e:
            logger.exception("Text extraction failed: %s", e)
            raise

        # sectionize
        sectioned = sectionaize_extracted_text(
            extraction_json, use_model_fallback=use_model_fallback
        )
        sectioned["raw_extraction"] = extraction_json

    response: Dict[str, Any] = {"status": "ok", "sectioned": sectioned}

    # Optionally run feature extraction (phase-3) if module present
    if run_feature_extraction:
        if extract_features_from_sectioned is None:
            logger.warning(
                "Requested feature extraction but extract_features_from_sectioned not available."
            )
            response["features"] = None
            response["feature_extraction_note"] = "module_not_available"
        else:
            try:
                logger.info("Running feature extraction on sectioned JSON")
                features = extract_features_from_sectioned(sectioned, jd_text=jd_text)
                response["features"] = features
            except Exception as e:
                logger.exception("Feature extraction failed: %s", e)
                response["features"] = None
                response["feature_extraction_error"] = str(e)
    else:
        response["features"] = None
        response["feature_extraction_note"] = "run_feature_extraction=False"

    return response
