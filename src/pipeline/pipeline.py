# src/pipeline/pipeline.py
"""
Pipeline orchestrator (updated).
This file provides helpers to run the pipeline end-to-end or partially.
It expects the feature extraction module to expose run_feature_extraction_from_sections.
It also attempts to use text_extraction and sectioning modules if present; otherwise,
the pipeline's `run_feature_extraction_from_sectioned` is the main entrypoint used by APIs.
"""

from typing import Dict, Any, Optional

# Import the feature-extraction facade. main.py (feature_extraction) provides run_feature_extraction_from_sections.
from src.phases.feature_extraction.main import run_feature_extraction_from_sections

# Try to import text_extraction & sectioning phases if they exist. If not present, API can pass sectioned JSON directly.
try:
    from src.phases.text_extraction.main import run_text_extraction_from_pdf
except Exception:
    run_text_extraction_from_pdf = None

try:
    from src.phases.sectioning.main import run_sectioning_on_text
except Exception:
    run_sectioning_on_text = None


def run_feature_extraction_from_sectioned(
    raw_sections: Dict[str, str],
    section_confidence: Optional[Dict[str, float]] = None,
    link_metadata: Optional[list] = None,
    emails: Optional[list] = None,
    phones: Optional[list] = None,
    jd_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the feature extraction phase directly on already-sectioned data.
    Useful for API endpoints that receive pre-sectioned payloads.
    Returns the same shape as the feature-extraction output.
    """
    return run_feature_extraction_from_sections(
        raw_sections,
        section_confidence=section_confidence,
        link_metadata=link_metadata,
        emails=emails,
        phones=phones,
        jd_text=jd_text,
    )


def run_pipeline_for_pdf(pdf_path: str, jd_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrator to run text_extraction -> sectioning -> feature_extraction,
    when the corresponding phase modules are present.
    Returns the final candidate JSON (feature extraction output).
    """
    if run_text_extraction_from_pdf is None or run_sectioning_on_text is None:
        raise RuntimeError(
            "Text extraction or sectioning modules are not available in pipeline context. "
            "To run the pipeline on PDF, ensure src.phases.text_extraction.main and src.phases.sectioning.main are present."
        )

    # Step 1: extract text (should return cleaned_text_updated, link_metadata, emails, phones)
    extracted = run_text_extraction_from_pdf(pdf_path)
    cleaned_text = extracted.get("cleaned_text_updated", "")
    link_metadata = extracted.get("link_metadata", [])
    emails = extracted.get("emails", [])
    phones = extracted.get("phones", [])

    # Step 2: sectioning -> returns raw_sections, section_confidence
    sectioned = run_sectioning_on_text(cleaned_text)
    raw_sections = sectioned.get("raw_sections", {})
    section_confidence = sectioned.get("section_confidence", {})

    # Step 3: feature extraction
    candidate = run_feature_extraction_from_sections(
        raw_sections,
        section_confidence=section_confidence,
        link_metadata=link_metadata,
        emails=emails,
        phones=phones,
        jd_text=jd_text,
    )
    return candidate
