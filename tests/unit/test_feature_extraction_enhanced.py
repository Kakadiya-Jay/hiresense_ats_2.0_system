# tests/unit/test_feature_extraction_enhanced.py
import json
from pathlib import Path

try:
    from src.phases.feature_extraction.main import run_feature_extraction_from_sections
except Exception:
    from ..src.phases.feature_extraction.main import run_feature_extraction_from_sections  # type: ignore

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_FILE_CANDIDATES = [
    ROOT / "data" / "sample_clean_text_response.txt",
    ROOT / "sample_clean_text_response.txt",
    ROOT / "tests" / "sample_payloads" / "sectioned_payload.json",
    Path.cwd() / "data" / "sample_clean_text_response.txt",
]


def find_sample():
    for p in CANDIDATE_FILE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find sample_clean_text_response.txt in expected locations."
    )


def load_sample():
    p = find_sample()
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    # attempt json parse
    try:
        parsed = json.loads(text)
        cleaned = parsed.get("cleaned_text_updated", text)
        link_metadata = parsed.get("link_metadata", [])
        emails = parsed.get("emails", [])
        phones = parsed.get("phones", [])
        return cleaned, link_metadata, emails, phones
    except Exception:
        return text, [], [], []


def test_enhanced_extractor_basic_fields():
    cleaned, link_metadata, emails, phones = load_sample()
    raw_sections = {"preamble": cleaned}
    candidate = run_feature_extraction_from_sections(
        raw_sections,
        section_confidence={"preamble": 0.9},
        link_metadata=link_metadata,
        emails=emails,
        phones=phones,
    )
    assert isinstance(candidate, dict)
    assert "candidate_id" in candidate
    assert "created_at" in candidate
    assert "personal_info" in candidate and isinstance(candidate["personal_info"], dict)
    # features present
    assert "features" in candidate and isinstance(candidate["features"], dict)
    # skills canonical list present
    skills = candidate["features"].get("skills", {})
    assert isinstance(skills, dict)
    # projects list (may be empty)
    projs = candidate["features"].get("projects", {}).get("projects", [])
    assert isinstance(projs, list)
    # emails propagation
    if emails:
        assert candidate.get("all_emails") and emails[0] in candidate.get("all_emails")
