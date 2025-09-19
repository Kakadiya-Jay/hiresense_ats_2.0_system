# tests/integration/test_process_resume_upload.py
"""
Integration test: post a multipart file to /process_resume.
Uses TestClient and the app entrypoint.
This test uses fake PDF bytes (no heavy PDF parsing).
"""

import io
from fastapi.testclient import TestClient
import pytest

# import FastAPI app
try:
    from src.api.main import app
except Exception:
    from api.main import app  # fallback

client = TestClient(app)


def test_process_resume_with_file_bytes_returns_candidate():
    # fake PDF bytes - not a real PDF, enough for our mock extractor
    fake_pdf = io.BytesIO(b"%PDF-1.4 \n Fake PDF content for tests\n")
    files = {"file": ("resume.pdf", fake_pdf, "application/pdf")}
    response = client.post("/process_resume", files=files)
    assert (
        response.status_code == 200
    ), f"status={response.status_code}, body={response.text}"
    data = response.json()
    assert "candidate" in data
    candidate = data["candidate"]
    # minimal assertions on candidate shape
    assert isinstance(candidate, dict)
    assert "candidate_id" in candidate
    assert "personal_info" in candidate
    assert isinstance(candidate.get("skills", []), list)
    assert isinstance(candidate.get("projects", []), list)


def test_process_resume_json_body_fallback():
    # fallback: send cleaned_text_updated JSON path
    payload = {
        "cleaned_text_updated": "Name: Test User\nSkills: Python, Java\nEducation: B.Tech",
        "emails": ["test@example.com"],
        "phones": ["+91 99999 99999"],
    }
    resp = client.post("/process_resume", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "candidate" in data
    assert data["candidate"]["personal_info"]["email"] == "test@example.com"
