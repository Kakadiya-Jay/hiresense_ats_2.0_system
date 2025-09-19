# src/phases/feature_extraction/main.py
"""
Feature extraction facade.

Exposes run_feature_extraction_from_sections(...) which uses the enhanced extractor.
"""

from .helpers.enhanced_extractor import build_candidate_from_sections


def run_feature_extraction_from_sections(
    raw_sections, section_confidence=None, link_metadata=None, emails=None, phones=None
):
    """
    Wrapper used by pipeline/service. Returns a candidate dict built from sections.
    """
    return build_candidate_from_sections(
        raw_sections,
        section_confidence=section_confidence,
        link_metadata=link_metadata,
        emails=emails,
        phones=phones,
    )
