# tests/unit/test_feature_extraction_from_cleaned_text.py
"""
Unit test: run feature extraction on the cleaned_text_response (the Postman extractor output).
This test loads `data/sample_clean_text_response.txt` (or repo-root sample file),
parses the JSON produced by your text extraction phase, and invokes the
feature-extraction facade using the cleaned text wrapped as a 'preamble' section.

Place sample_clean_text_response.txt in the repo root or update SAMPLE_PATH accordingly.
"""

import json
from pathlib import Path

try:
    # preferred import path when running tests from repo root
    from src.phases.feature_extraction.main import run_feature_extraction_from_sections
except Exception:
    # fallback import if test runner's PYTHONPATH differs
    from ..src.phases.feature_extraction.main import run_feature_extraction_from_sections  # type: ignore


def find_sample_file():
    cwd_guess = Path.cwd() / "data/sample_clean_text_response.txt"
    if cwd_guess.exists():
        return cwd_guess
    raise FileNotFoundError(
        f"Could not find sample_clean_text_response.txt in any expected location: {cwd_guess}"
    )


def load_sample_clean_text():
    p = find_sample_file()
    text = p.read_text(encoding="utf-8").strip()
    # file could contain raw JSON object (single line) or plain text.
    try:
        parsed = json.loads(text)
        # expecting keys similar to your extractor response
        cleaned = parsed.get("cleaned_text_updated", "")
        link_metadata = parsed.get("link_metadata", []) or []
        emails = parsed.get("emails", []) or []
        phones = parsed.get("phones", []) or []
        return cleaned, link_metadata, emails, phones
    except Exception:
        # fallback: whole file as cleaned text
        return text, [], [], []


def test_feature_extraction_from_cleaned_text_creates_candidate():
    cleaned_text, link_metadata, emails, phones = load_sample_clean_text()
    # create a minimal sectioned payload by wrapping cleaned_text as preamble
    raw_sections = {"preamble": cleaned_text}
    section_confidence = {"preamble": 0.9}

    candidate = run_feature_extraction_from_sections(
        raw_sections,
        section_confidence=section_confidence,
        link_metadata=link_metadata,
        emails=emails,
        phones=phones,
    )

    # Basic shape checks
    assert isinstance(candidate, dict), "candidate must be a dict"
    assert (
        "candidate_id" in candidate and candidate["candidate_id"]
    ), "candidate_id must be present"
    assert (
        "created_at" in candidate and candidate["created_at"]
    ), "created_at must be present"
    assert "personal_info" in candidate and isinstance(candidate["personal_info"], dict)
    assert "skills" in candidate and isinstance(candidate["skills"], list)
    assert "projects" in candidate and isinstance(candidate["projects"], list)
    assert "education" in candidate and isinstance(candidate["education"], list)
    assert "raw_sections" in candidate and isinstance(candidate["raw_sections"], dict)

    # Email/phone propagation (if present in sample)
    if emails:
        # extractor should propagate the first email into personal_info.email
        assert candidate["personal_info"].get("email") == emails[0]
        assert emails[0] in candidate.get("all_emails", [])
    if phones:
        # if phones passed, personal_info.phone should equal the first phone (we pass raw phone list)
        assert candidate["personal_info"].get("phone") == phones[0]
        assert phones[0] in candidate.get("all_phones", [])

    # sanity checks on lists
    assert len(candidate["skills"]) <= 1000  # sanity upper bound
    # projects entries should have minimal keys if any exist
    for proj in candidate.get("projects", []):
        assert isinstance(proj, dict)
        assert "title" in proj and "summary" in proj and "tech_stack" in proj
