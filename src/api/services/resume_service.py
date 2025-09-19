# src/api/services/resume_service.py
"""
Resume service wrapper (production-ready).

Responsibilities:
- Accept payload dicts coming from API layer and/or raw PDF bytes.
- If `file_bytes` present: call text_extraction -> sectioning -> feature_extraction pipeline.
- If `raw_sections` present: call feature_extraction directly.
- If only `cleaned_text_updated` present: wrap into a 'preamble' section and call feature_extraction.
- Accepts optional base64-encoded file bytes in payload via 'file_bytes_b64'.
- Returns a candidate dict (feature extraction output) or raises ValueError on bad input.

Notes for production:
- Ensure your text_extraction and sectioning facades (modules/functions) are available
  at the import paths used below. If your project uses different module paths,
  adjust the imports.
- Avoid sending raw bytes in JSON; prefer multipart UploadFile to the FastAPI route,
  then pass bytes to this service (file_bytes param).
"""

import base64
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import feature extraction facade
try:
    # Preferred absolute import (adjustable to your project layout)
    from src.phases.feature_extraction.main import run_feature_extraction_from_sections
except Exception:
    # fallback to relative import if running under different PYTHONPATH
    from ..phases.feature_extraction.main import run_feature_extraction_from_sections  # type: ignore

# Try to import text_extraction and sectioning facades if they exist.
# These are optional if you only call this service with sectioned payloads.
_text_extraction_available = False
_sectioning_available = False
try:
    from src.phases.text_extraction.main import run_text_extraction_from_pdf_bytes  # type: ignore

    _text_extraction_available = True
except Exception:
    try:
        # fallback relative import
        from ..phases.text_extraction.main import run_text_extraction_from_pdf_bytes  # type: ignore

        _text_extraction_available = True
    except Exception:
        _text_extraction_available = False

try:
    from src.phases.sectioning.main import run_sectioning_on_text  # type: ignore

    _sectioning_available = True
except Exception:
    try:
        from ..phases.sectioning.main import run_sectioning_on_text  # type: ignore

        _sectioning_available = True
    except Exception:
        _sectioning_available = False


def _decode_b64_if_present(payload: Dict[str, Any]) -> Optional[bytes]:
    """
    If payload contains 'file_bytes_b64' (base64 string), decode and return bytes.
    Otherwise return None.
    """
    b64 = payload.get("file_bytes_b64")
    if not b64:
        return None
    try:
        # if payload includes data URI prefix, strip it
        if isinstance(b64, str) and b64.startswith("data:"):
            # split on first comma
            parts = b64.split(",", 1)
            if len(parts) == 2:
                b64 = parts[1]
        return base64.b64decode(b64)
    except Exception as ex:
        logger.warning("Failed to decode base64 file bytes: %s", ex)
        return None


def process_resume_payload(
    payload: Dict[str, Any], file_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    """
    Main service function.

    Parameters:
        payload: dict (may contain raw_sections / cleaned_text_updated / emails / phones / link_metadata)
        file_bytes: optional bytes object (raw PDF bytes). If provided, service will run extraction & sectioning.

    Returns:
        candidate dict (as produced by feature extraction)

    Raises:
        ValueError on bad input / missing required pieces.
    """
    # defensive checks & normalization
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dict.")

    # If file_bytes not provided explicitly, check if payload contains base64 field
    if file_bytes is None:
        maybe = _decode_b64_if_present(payload)
        if maybe:
            file_bytes = maybe

    # If file_bytes is provided: run text_extraction -> sectioning -> feature_extraction
    if file_bytes:
        if not _text_extraction_available or not _sectioning_available:
            msg = "Text extraction or sectioning module is not available. Ensure src.phases.text_extraction.main.run_text_extraction_from_pdf_bytes and src.phases.sectioning.main.run_sectioning_on_text exist."
            logger.error(msg)
            raise ValueError(msg)

        try:
            # Step 1: extract text + metadata
            extracted = run_text_extraction_from_pdf_bytes(file_bytes)
            if not isinstance(extracted, dict):
                raise ValueError(
                    "Text extraction facade returned invalid result; expected dict."
                )
            cleaned_text = extracted.get("cleaned_text_updated", "")
            link_metadata = (
                extracted.get("link_metadata", [])
                or payload.get("link_metadata", [])
                or []
            )
            emails = extracted.get("emails", []) or payload.get("emails", []) or []
            phones = extracted.get("phones", []) or payload.get("phones", []) or []

            # Step 2: sectioning
            sectioned = run_sectioning_on_text(cleaned_text)
            if not isinstance(sectioned, dict):
                raise ValueError(
                    "Sectioning facade returned invalid result; expected dict."
                )
            raw_sections = sectioned.get("raw_sections", {}) or {}
            section_confidence = sectioned.get("section_confidence", {}) or {}

            # Step 3: feature extraction
            candidate = run_feature_extraction_from_sections(
                raw_sections,
                section_confidence=section_confidence,
                link_metadata=link_metadata,
                emails=emails,
                phones=phones,
            )
            return candidate
        except ValueError:
            # re-raise ValueError to caller
            raise
        except Exception as ex:
            logger.exception("Failed during process_resume_payload(file_bytes): %s", ex)
            raise ValueError(f"Internal error during file processing: {ex}") from ex

    # No file bytes path: prefer raw_sections if present
    raw_sections = payload.get("raw_sections")
    section_confidence = payload.get("section_confidence", {}) or {}
    link_metadata = (
        payload.get("link_metadata", []) or payload.get("link_meta", []) or []
    )
    emails = payload.get("emails", []) or []
    phones = payload.get("phones", []) or []

    if raw_sections and isinstance(raw_sections, dict):
        try:
            candidate = run_feature_extraction_from_sections(
                raw_sections,
                section_confidence=section_confidence,
                link_metadata=link_metadata,
                emails=emails,
                phones=phones,
            )
            return candidate
        except Exception as ex:
            logger.exception(
                "Feature extraction failed when called with raw_sections: %s", ex
            )
            raise ValueError(f"Feature extraction failed: {ex}") from ex

    # fallback: cleaned_text_updated path
    cleaned_text = payload.get("cleaned_text_updated", "")
    if cleaned_text and isinstance(cleaned_text, str):
        try:
            # Wrap as preamble section and call feature extractor
            sections = {"preamble": cleaned_text}
            candidate = run_feature_extraction_from_sections(
                sections,
                section_confidence=section_confidence,
                link_metadata=link_metadata,
                emails=emails,
                phones=phones,
            )
            return candidate
        except Exception as ex:
            logger.exception(
                "Feature extraction failed when called with cleaned_text_updated: %s",
                ex,
            )
            raise ValueError(f"Feature extraction failed: {ex}") from ex

    # Nothing to do
    raise ValueError(
        "Payload did not contain 'raw_sections' or 'cleaned_text_updated', and no file bytes were provided."
    )
