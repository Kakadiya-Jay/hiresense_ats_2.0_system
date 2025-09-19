# tests/unit/test_integration_sectioning.py
import json
from src.phases.sectioning.main import sectionize, build_candidate_skeleton


def test_sectionize_and_build_candidate_simple():
    # Build a minimal Phase1 meta with a single page text (fallback path)
    meta = {
        "pages": [
            "John Doe\njohn.doe@example.com\n\nPROFILE\nExperienced backend engineer\n\nSKILLS\nPython, Django\nPROJECTS\n- Resume Parser: Built a parser. Tech: Python, Regex\n"
        ],
        "candidate_id": "test001",
        "link_metadata": [],
    }
    sections = sectionize(meta)
    assert isinstance(sections, list) and len(sections) > 0
    cand = build_candidate_skeleton(sections, meta)
    assert cand.get("candidate_id") == "test001"
    # personal_info name or email should be present
    assert "personal_info" in cand
    assert cand["personal_info"].get("email") or cand["personal_info"].get("name")
