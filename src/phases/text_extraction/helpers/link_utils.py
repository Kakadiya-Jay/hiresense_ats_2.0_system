# src/phases/text_extraction/helpers/link_utils.py
"""
URL normalization and deduplication helpers.
Public:
- normalize_url(url) -> str
- dedupe_links(links) -> List[dict]  (expects links with 'url','anchor_text','page')
"""

import urllib.parse
from typing import List, Dict

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}


def normalize_url(url: str) -> str:
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    if url.lower().startswith("mailto:"):
        # return email address only
        return url.split(":", 1)[1].strip().lower()
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
    Deduplicate by normalized_url and merge anchor_text / original_urls / page info.
    Returns list of entries:
      { normalized_url, anchor_text, page, original_urls (list) }
    """
    by_norm = {}
    for l in links:
        raw = l.get("url") or ""
        anchor = (l.get("anchor_text") or "").strip()
        norm = normalize_url(raw)
        if not norm:
            continue
        if norm in by_norm:
            prev = by_norm[norm]
            if (not prev.get("anchor_text")) and anchor:
                prev["anchor_text"] = anchor
            prev["original_urls"].add(raw)
            if prev.get("page") is None and l.get("page") is not None:
                prev["page"] = l.get("page")
        else:
            by_norm[norm] = {
                "normalized_url": norm,
                "anchor_text": anchor,
                "page": l.get("page"),
                "original_urls": {raw},
            }
    out = []
    for v in by_norm.values():
        v["original_urls"] = list(v["original_urls"])
        out.append(v)
    return out
