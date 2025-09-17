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

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\)\]\,;]+", flags=re.I)
ARXIV_ID_RE = re.compile(r"([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", flags=re.I)

PUBLISHER_DOMAINS = [
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "springer.com",
    "nature.com",
    "sciencedirect.com",
    "researchgate.net",
    "pubmed.ncbi.nlm.nih.gov",
    "ssrn.com",
]


def classify_link(entry: Dict) -> Dict:
    """
    Input: entry with 'normalized_url', 'anchor_text', 'page', 'original_urls'
    Returns the entry augmented with: domain, type, meta, confidence
    """
    norm = entry.get("normalized_url", "") or ""
    anchor = (entry.get("anchor_text") or "").lower()
    try:
        domain = (
            urllib.parse.urlparse(norm).netloc.lower()
            if norm.startswith("http")
            else ""
        )
    except Exception:
        domain = ""
    out = dict(entry)
    out["domain"] = domain
    out["type"] = "other"
    out["meta"] = {}
    out["confidence"] = 0.5

    # ---- Research / paper detection ----
    # 1) arXiv direct (high confidence)
    if "arxiv.org" in norm:
        # extract id
        m = re.search(
            r"/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", norm, flags=re.I
        )
        if m:
            arxiv_id = m.group(1)
            out["type"] = "research_paper"
            out["meta"] = {"source": "arxiv", "id": arxiv_id}
            out["confidence"] = 0.95
            return out
        # if no id found, still tag as research candidate
        out["type"] = "research_paper"
        out["meta"] = {"source": "arxiv", "id": None}
        out["confidence"] = 0.85
        return out

    # 2) DOI normalized "doi:" or doi.org links
    if norm.startswith("doi:") or "doi.org" in norm:
        # extract doi id portion
        doi = None
        if norm.startswith("doi:"):
            doi = norm.split(":", 1)[1]
        else:
            try:
                doi = urllib.parse.urlparse(norm).path.lstrip("/")
            except Exception:
                doi = None
        out["type"] = "research_paper"
        out["meta"] = {"source": "doi", "id": doi}
        out["confidence"] = 0.95
        return out

    # 3) Publisher domains (ACM, IEEE, Springer...) -> paper-like
    for pub in PUBLISHER_DOMAINS:
        if pub in norm:
            out["type"] = "research_paper"
            out["meta"] = {"source": pub.split(".")[0], "id": None}
            out["confidence"] = 0.90
            return out

    # 4) PDF file heuristics: url ends with .pdf (medium confidence)
    if norm.lower().endswith(".pdf"):
        # increase confidence if anchor or surrounding anchor indicates 'paper' or 'publication'
        if re.search(
            r"\b(paper|preprint|pdf|publication|accepted|conference|journal)\b",
            anchor,
            flags=re.I,
        ):
            out["type"] = "research_paper"
            out["meta"] = {"source": "pdf", "id": None}
            out["confidence"] = 0.85
            return out
        # otherwise mark as file, but still possibly research if in publications later
        out["type"] = "file"
        out["meta"] = {"source": "pdf", "id": None}
        out["confidence"] = 0.6
        return out

    # 5) Anchor heuristics: anchor text mentions pdf/doi/arxiv/paper/publication
    if re.search(
        r"\b(arxiv|doi|paper|pdf|preprint|publication|accepted|conference|journal)\b",
        anchor,
        flags=re.I,
    ):
        out["type"] = "research_paper"
        out["meta"] = {"source": "anchor_hint", "id": None}
        out["confidence"] = 0.75
        return out

    # ---- Existing heuristics for code repos / linkedin / certificates ----
    # github
    if "github.com" in domain:
        path = urllib.parse.urlparse(norm).path.strip("/")
        segs = [s for s in path.split("/") if s]
        if len(segs) >= 2:
            out["type"] = "github_repo"
            out["meta"] = {"user": segs[0].lower(), "repo": segs[1].lower()}
            out["confidence"] = 0.98
            return out
        elif len(segs) == 1 and segs[0]:
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

    # anchor heuristics for other types
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
    # ensure each has normalized_url (dedupe already provided)
    for e in deduped:
        e["normalized_url"] = e.get("normalized_url") or e.get("normalized_url")
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
    cleaned_updated = inject_links_into_text(text, link_metadata)
    m["cleaned_text_updated"] = cleaned_updated
    m["link_metadata"] = link_metadata
    return m
