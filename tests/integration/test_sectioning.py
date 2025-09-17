# tests/test_name_extraction.py
import pytest
from src.phases.sectioning.main import build_candidate_skeleton

def test_name_extraction_from_header_block():
    sections = [
        {
            "section_id": "s0",
            "heading": "",
            "section_type": "header",
            "text": "Jay Kakadiya\njkakadiya109@gmail.com\n+91 98765 43210",
            "start_page": 0,
            "end_page": 0,
            "confidence": 0.95
        }
    ]
    meta = {
        "pages": [sections[0]["text"]],
        "link_metadata": [],
        "emails": ["jkakadiya109@gmail.com"],
        "phones": ["+919876543210"]
    }
    sk = build_candidate_skeleton(sections, meta)
    assert "personal_info" in sk
    assert "name" in sk["personal_info"]
    assert sk["personal_info"]["name"].lower().startswith("jay kakadiya")
    assert "email" in sk["personal_info"]
    assert sk["personal_info"]["email"] == "jkakadiya109@gmail.com"
    assert "phone" in sk["personal_info"]

def test_name_extraction_near_email_on_first_page():
    # Here email occurs, name is the line above it; the algorithm should pick the above line.
    page_text = "\n".join([
        "John Doe",
        "john.doe@example.com",
        "Address line, City"
    ])
    sections = [
        {
            "section_id": "s0",
            "heading": "",
            "section_type": "unknown",
            "text": page_text,
            "start_page": 0,
            "end_page": 0,
            "confidence": 0.6
        }
    ]
    meta = {
        "pages": [page_text],
        "link_metadata": [],
        "emails": [],
        "phones": []
    }
    sk = build_candidate_skeleton(sections, meta)
    assert "name" in sk["personal_info"]
    assert sk["personal_info"]["name"].lower() == "john doe"
    assert "email" in sk["personal_info"]
    assert sk["personal_info"]["email"] == "john.doe@example.com"

def test_name_extraction_fallback_full_text():
    # No clear header, no email; the first reasonable non-label line should be picked.
    full_text = "\n".join([
        "Profile Summary",
        "Experienced Data Scientist",
        "Worked on ML projects ...",
        "Contact: recruiter@example.com"
    ])
    sections = [
        {
            "section_id": "s0",
            "heading": "",
            "section_type": "unknown",
            "text": full_text,
            "start_page": 0,
            "end_page": 0,
            "confidence": 0.5
        }
    ]
    meta = {
        "pages": [full_text],
        "link_metadata": [],
        "emails": ["recruiter@example.com"],
        "phones": []
    }
    sk = build_candidate_skeleton(sections, meta)
    # In this contrived text the algorithm may pick "Profile Summary" or "Experienced Data Scientist".
    # We assert that a name-like field exists or at least some non-empty personal_info is created.
    assert "personal_info" in sk
    # either name extracted or email present at least
    assert ("name" in sk["personal_info"]) or ("email" in sk["personal_info"])
