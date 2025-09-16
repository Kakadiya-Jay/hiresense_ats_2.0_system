# src/phases/text_extraction/helpers/classifier.py
"""
Classify normalized links into types and inject them into text.
Public:
- classify_link(entry) -> dict (adds type, domain, meta, confidence)
- inject_links_into_text(text, link_meta) -> str
- update_extract_and_clean(meta) -> meta with cleaned_text_updated and link_metadata
"""

import re
import urllib.parse
from typing import List, Dict
from .link_utils import dedupe_links, normalize_url

CERT_DOMAINS = {"codebasics.io", "coursera.org", "udemy.com", "edx.org", "nptel.ac.in"}

DOI_RE = re.compile(r"10\.\d{4,9}/\S+", flags=re.IGNORECASE)


def classify_link(entry: Dict) -> Dict:
    """
    Input: entry with 'normalized_url', 'anchor_text', 'page', 'original_urls'
    Returns the entry augmented with: domain, type, meta, confidence
    """
    norm = entry.get("normalized_url", "") or ""
    anchor = (entry.get("anchor_text") or "").lower()
    try:
        domain = urllib.parse.urlparse(norm).netloc.lower()
    except Exception:
        domain = ""
    out = dict(entry)
    out["domain"] = domain
    out["type"] = "other"
    out["meta"] = {}
    out["confidence"] = 0.5

    # github
    if "github.com" in domain:
        path = urllib.parse.urlparse(norm).path.strip("/")
        segs = [s for s in path.split("/") if s]
        if len(segs) >= 2:
            out["type"] = "github_repo"
            out["meta"] = {"user": segs[0].lower(), "repo": segs[1].lower()}
            out["confidence"] = 0.98
            return out
        if len(segs) == 1 and segs[0]:
            out["type"] = "github_profile"
            out["meta"] = {"user": segs[0].lower()}
            out["confidence"] = 0.95
            return out

    # linkedin
    if "linkedin.com" in domain:
        if "/details/certifications/" in norm:
            out["type"] = "linkedin_cert"
            out["confidence"] = 0.92
            return out
        if "/in/" in norm or "/pub/" in norm:
            out["type"] = "linkedin_profile"
            out["confidence"] = 0.95
            return out

    # certificates
    if any(d in domain for d in CERT_DOMAINS):
        out["type"] = "certificate"
        out["confidence"] = 0.9
        return out

    # research
    if "arxiv.org" in domain or DOI_RE.search(norm) or DOI_RE.search(anchor):
        out["type"] = "research_paper"
        out["confidence"] = 0.95
        return out

    # anchor heuristics
    if re.search(r"\b(repo|repository|source|github)\b", anchor):
        out["type"] = "github_repo"
        out["confidence"] = 0.7
        return out
    if re.search(r"\b(cert|certificate|certification)\b", anchor):
        out["type"] = "certificate"
        out["confidence"] = 0.7
        return out
    if re.search(r"\blinkedin\b", anchor):
        out["type"] = "linkedin_profile"
        out["confidence"] = 0.7
        return out

    return out


def inject_links_into_text(text: str, link_meta: List[Dict]) -> str:
    """Insert normalized_url next to anchor_text where present; otherwise append at end."""
    out = text
    for l in link_meta:
        anchor = l.get("anchor_text") or ""
        url = l.get("normalized_url")
        if not url:
            continue
        if anchor and anchor.lower() in out.lower():
            out = re.sub(re.escape(anchor), f"{anchor} {url}", out, count=1, flags=re.I)
        else:
            if url not in out:
                out = out + " " + url
    out = re.sub(r"\s+", " ", out).strip()
    return out


def update_extract_and_clean(meta: Dict) -> Dict:
    """
    Input meta from extractor (full_text, cleaned_text, links, emails, phones).
    Normalizes/dedupes/classifies links, injects normalized urls into cleaned_text,
    and returns updated meta with:
      - cleaned_text_updated
      - link_metadata (list of classified items)
    """
    m = dict(meta)
    text = m.get("cleaned_text") or m.get("full_text") or ""
    raw_links = m.get("links", [])

    # prepare normalized, deduped entries (add normalized_url field)
    deduped = dedupe_links(raw_links)
    for e in deduped:
        e["normalized_url"] = e.get("normalized_url") or e.get(
            "normalized_url"
        )  # already present from dedupe
    # classify
    classified = [classify_link(e) for e in deduped]
    # build metadata schema
    link_metadata = []
    for c in classified:
        item = {
            "normalized_url": c.get("normalized_url"),
            "original_urls": c.get("original_urls", []),
            "domain": c.get("domain"),
            "anchor_text": c.get("anchor_text"),
            "page": c.get("page"),
            "type": c.get("type"),
            "meta": c.get("meta", {}),
            "confidence": float(c.get("confidence", 0.5)),
        }
        link_metadata.append(item)
    # inject urls into text
    cleaned_updated = inject_links_into_text(text, link_metadata)
    m["cleaned_text_updated"] = cleaned_updated
    m["link_metadata"] = link_metadata
    return m
