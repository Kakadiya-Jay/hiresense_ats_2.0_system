# src/phases/sectioning/helpers/name_utils.py
"""
Fallback name inference utilities used by Sectioning (Phase-2).
Attempts multiple heuristics if extraction_json['candidate_name'] is missing or empty.
"""

import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"([^@]+)@")
_LINKEDIN_RE = re.compile(
    r"(?:linkedin\.com/(?:in|pub)/)([A-Za-z0-9\-_%\.]+)", re.IGNORECASE
)


def _titlecase_name_frag(s: str) -> str:
    parts = [p for p in re.split(r"[^\w']+", s.strip()) if p]
    # take first 2-3 tokens
    tokens = parts[:3]
    return " ".join([t.capitalize() for t in tokens])


def infer_from_email(extraction: Dict[str, Any]) -> Optional[str]:
    emails = extraction.get("emails") or extraction.get("email") or []
    if isinstance(emails, list) and emails:
        m = _EMAIL_RE.search(emails[0])
        if m:
            username = m.group(1)
            # split on dots or digits/hyphens and choose plausible tokens
            # e.g. 'jkakadiya109' -> 'jkakadiya' -> 'Jkakadiya' -> 'J K' not ideal,
            # but we'll attempt dot/underscore split first
            parts = re.split(r"[._\-0-9]+", username)
            parts = [p for p in parts if len(p) > 1]
            if parts:
                name_guess = " ".join([p.capitalize() for p in parts[:2]])
                logger.debug("Inferred candidate_name from email: %s", name_guess)
                return name_guess
    return None


def infer_from_link_metadata(extraction: Dict[str, Any]) -> Optional[str]:
    links = extraction.get("link_metadata") or []
    for l in links:
        url = l.get("normalized_url") or l.get("url") or ""
        if not url:
            continue
        m = _LINKEDIN_RE.search(url)
        if m:
            username = m.group(1)
            # LinkedIn usernames often are like 'jay-kakadiya-96647625b' -> take first two tokens
            tokens = re.split(r"[-_.]+", username)
            tokens = [t for t in tokens if t and not re.fullmatch(r"\d+", t)]
            if tokens:
                name_guess = " ".join([t.capitalize() for t in tokens[:2]])
                logger.debug(
                    "Inferred candidate_name from LinkedIn URL: %s", name_guess
                )
                return name_guess
    return None


def infer_from_raw_text(extraction: Dict[str, Any]) -> Optional[str]:
    # Use cleaned_text_updated or raw_text; prefer page[0] top lines if available.
    text = extraction.get("cleaned_text_updated") or extraction.get("raw_text") or ""
    if not text:
        return None
    # examine first ~6 lines for a short candidate-looking line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head_lines = lines[:8]
    for ln in head_lines:
        # skip lines that contain email/phone or URLs
        if "@" in ln or re.search(r"\+?\d{6,}", ln) or "http" in ln.lower():
            continue
        # consider lines with 2-4 words and alphabetic initials
        tokens = [t for t in re.split(r"\s+", ln) if t]
        if 2 <= len(tokens) <= 4:
            # ensure tokens start with uppercase letter or are all-caps name style
            if all(re.match(r"^[A-Za-z][a-zA-Z\.'\-]+$", t) for t in tokens):
                name_guess = " ".join([t.strip() for t in tokens])
                logger.debug("Inferred candidate_name from top lines: %s", name_guess)
                return name_guess
            # fallback: titlecase the small line
            name_guess = _titlecase_name_frag(ln)
            if len(name_guess.split()) >= 2:
                logger.debug(
                    "Fallback inferred candidate_name from top lines (titlecase): %s",
                    name_guess,
                )
                return name_guess
    return None


def infer_candidate_name(extraction: Dict[str, Any]) -> Optional[str]:
    """
    Return the best candidate_name inferred either from extraction['candidate_name']
    or from heuristics. Returns None if not found.
    """
    # 1) direct value if present and non-empty
    cn = extraction.get("candidate_name")
    if cn and isinstance(cn, str) and cn.strip():
        return cn.strip()

    # 2) try link metadata (LinkedIn)
    from_link = infer_from_link_metadata(extraction)
    if from_link:
        return from_link

    # 3) try email username
    from_email = infer_from_email(extraction)
    if from_email:
        return from_email

    # 4) try raw text top lines
    from_text = infer_from_raw_text(extraction)
    if from_text:
        return from_text

    # no candidate name found
    return None
