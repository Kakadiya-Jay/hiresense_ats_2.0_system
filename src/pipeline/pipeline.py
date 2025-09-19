# src/pipeline/pipeline.py
"""
Pipeline orchestrator (updated).
This file provides helpers to run the pipeline end-to-end or partially.
We add `run_feature_extraction_from_sectioned` to allow the API/service to pass
sectioned data (produced earlier by sectioning) directly to the feature extractor.

If you already have a run_pipeline_for_pdf implementation, merge these helpers with it.
"""

from typing import Dict, Any, Optional

# Imports of phase facades (assumes these modules exist in the repo)
# text_extraction.main -> hypothetically provides `extract_text_from_pdf`
# sectioning.main -> hypothetically provides `run_sectioning_on_text`
# we import feature extraction function we added above
from src.phases.feature_extraction.main import run_feature_extraction_from_sections

# Optional imports of other phases (if you have these, pipeline can call them)
# Try to import text_extraction and sectioning facades if present.
from src.phases.text_extraction.main import run_text_extraction_from_pdf
from src.phases.sectioning.main import run_sectioning_on_text

def run_feature_extraction_from_sectioned(
    raw_sections: Dict[str, str],
    section_confidence: Optional[Dict[str, float]] = None,
    link_metadata: Optional[list] = None,
    emails: Optional[list] = None,
    phones: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Run the feature extraction phase directly on already-sectioned data.
    This is useful when the text extraction + sectioning have already run
    (e.g., you receive that payload from a client or postman).
    """
    return run_feature_extraction_from_sections(
        raw_sections,
        section_confidence=section_confidence,
        link_metadata=link_metadata,
        emails=emails,
        phones=phones,
    )


def run_pipeline_for_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Example orchestrator: run text_extraction -> sectioning -> feature_extraction.
    This is a helpful reference function; adapt as per your existing pipeline implementations.
    """
    if run_text_extraction_from_pdf is None or run_sectioning_on_text is None:
        raise RuntimeError(
            "Text extraction or sectioning modules are not available in pipeline context."
        )

    # Step 1: extract text + link metadata + emails/phones
    extracted = run_text_extraction_from_pdf(pdf_path)
    cleaned_text = extracted.get("cleaned_text_updated", "")
    link_metadata = extracted.get("link_metadata", [])
    emails = extracted.get("emails", [])
    phones = extracted.get("phones", [])

    # Step 2: sectioning
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
    )
    return candidate
