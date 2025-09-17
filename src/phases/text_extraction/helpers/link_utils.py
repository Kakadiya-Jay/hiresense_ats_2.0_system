# src/phases/text_extraction/helpers/link_utils.py
"""
URL normalization and deduplication helpers.
Public:
- normalize_url(url) -> str
- dedupe_links(links) -> List[dict]  (expects links with 'url','anchor_text','page')
"""

import urllib.parse
from typing import List, Dict
import re

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}

DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\)\]\,;]+)", flags=re.I)


def normalize_url(url: str) -> str:
    """
    Normalize urls and DOI strings. If `url` contains a DOI and is not a regular http(s) URL,
    return 'doi:<DOI>' (lowercased DOI). Otherwise, canonicalize HTTP URLs.
    """
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    # detect mailto
    if url.lower().startswith("mailto:"):
        return url.split(":", 1)[1].strip().lower()
    # detect plain DOI forms like "10.1145/1234.5678"
    m = DOI_RE.search(url)
    if m and not url.lower().startswith("http"):
        doi = m.group(1).strip().lower()
        return f"doi:{doi}"
    # if DOI present in http URL, canonicalize to doi: form if it is doi.org
    if "doi.org" in url.lower():
        try:
            p = urllib.parse.urlparse(url)
            doi = p.path.lstrip("/")
            return f"doi:{doi.lower()}"
        except Exception:
            pass
    if not url.startswith("http"):
        url = "https://" + url
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return url
    scheme = p.scheme.lower()
    netloc = p.netloc.lower()
    path = urllib.parse.unquote(p.path or "").rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    qs = urllib.parse.parse_qs(p.query or "")
    qs = {k: v for k, v in qs.items() if k not in TRACKING_PARAMS}
    query = urllib.parse.urlencode(qs, doseq=True)
    normalized = urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))
    return normalized.rstrip("/")


def dedupe_links(links: List[Dict]) -> List[Dict]:
    """
    Deduplicate by normalized_url and by id when present (doi/arxiv id).
    Returns list of entries:
      { normalized_url, anchor_text, page, original_urls (list), meta_id (optional) }
    """
    by_key = {}
    for l in links:
        raw = l.get("url") or ""
        anchor = (l.get("anchor_text") or "").strip()
        norm = normalize_url(raw)
        if not norm:
            continue
        # attempt to use a key: if DOI prefix, use DOI id; if arxiv in path, use arxiv id; else use normalized url
        key = norm
        # DOI key
        if norm.startswith("doi:"):
            key = norm  # doi:<id>
        # arxiv heuristic
        if "arxiv.org" in norm:
            # try to extract id
            m = re.search(
                r"/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", norm, flags=re.I
            )
            if m:
                key = f"arxiv:{m.group(1)}".lower()
        if key in by_key:
            prev = by_key[key]
            if (not prev.get("anchor_text")) and anchor:
                prev["anchor_text"] = anchor
            prev["original_urls"].add(raw)
            if prev.get("page") is None and l.get("page") is not None:
                prev["page"] = l.get("page")
        else:
            by_key[key] = {
                "normalized_url": norm,
                "anchor_text": anchor,
                "page": l.get("page"),
                "original_urls": {raw},
                "meta_id": key if key != norm else None,
            }
    out = []
    for v in by_key.values():
        v["original_urls"] = list(v["original_urls"])
        out.append(v)
    return out
