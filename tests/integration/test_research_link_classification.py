# tests/test_research_link_classification.py
import re
from src.phases.text_extraction.helpers.classifier import classify_link
from src.phases.text_extraction.helpers.link_utils import normalize_url


def test_arxiv_detection():
    url = "https://arxiv.org/abs/2101.12345"
    norm = normalize_url(url)
    entry = {
        "normalized_url": norm,
        "anchor_text": "paper",
        "page": 0,
        "original_urls": [url],
    }
    classified = classify_link(entry)
    assert classified["type"] == "research_paper"
    assert classified["meta"].get("source") == "arxiv"
    assert "2101.12345" in (classified["meta"].get("id") or "")


def test_doi_detection():
    url = "10.1145/1234567.1234568"
    norm = normalize_url(url)
    # normalize_url should produce doi: prefix
    assert norm.startswith("doi:")
    entry = {
        "normalized_url": norm,
        "anchor_text": "",
        "page": 0,
        "original_urls": [url],
    }
    classified = classify_link(entry)
    assert classified["type"] == "research_paper"
    assert classified["meta"].get("source") == "doi"
    assert classified["meta"].get("id") == "10.1145/1234567.1234568"
