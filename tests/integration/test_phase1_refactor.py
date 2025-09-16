# tests/test_phase1_refactor.py
import os
import pytest
from src.phases.text_extraction.helpers import extractor, link_utils, classifier
from src.phases.text_extraction.main import run_text_extraction_pipeline
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

SAMPLE_PDF = "data/raw_resumes/Jay_Kakadiya_Fresher_Resume_for_placement.pdf"  # optional

def test_hyphen_fix():
    s = "optimi-\n zation example"
    out = extractor.fix_hyphenation_and_linebreaks(s)
    assert "optimi-" not in out

def test_normalize_and_dedupe_basic():
    links = [
        {"url": "https://github.com/Kakadiya-Jay/AdBroker.git", "anchor_text":"repo","page":0},
        {"url": "http://github.com/kakadiyajay/AdBroker", "anchor_text":"", "page": None},
        {"url": "MAILTO:JKAKADIYA109@GMAIL.COM", "anchor_text":"email", "page": 0}
    ]
    deduped = link_utils.dedupe_links(links)
    normalized = [d["normalized_url"] for d in deduped]
    assert any("github.com" in u for u in normalized)
    assert any("gmail.com" in u for u in normalized)

@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF), reason="sample pdf missing")
def test_process_resume_endpoint():
    with open(SAMPLE_PDF, "rb") as f:
        files = {"file": ("sample1.pdf", f, "application/pdf")}
        resp = client.post("/process_resume", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "cleaned_text_updated" in data
    assert "link_metadata" in data
