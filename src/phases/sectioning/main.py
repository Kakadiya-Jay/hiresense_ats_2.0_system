# src/phases/sectioning/main.py
"""
Sectioning module (rule-based baseline).

Primary function:
    run_sectioning_on_text(cleaned_text: str) -> dict

Behavior:
- Uses conservative header detection heuristics:
    - Known header keywords (education, experience, skills, projects, certifications, achievements, summary, languages)
    - Lines in ALL CAPS (short)
    - Lines ending with ':' or starting with header keywords
- Splits cleaned_text into sections and returns:
    - raw_sections: dict of canonical_section_name -> text_block
    - section_confidence: dict of canonical_section_name -> float [0..1]
- This is a rule-based baseline; replace with a model-based sectioner later for improved recall.
"""

import re
from typing import Dict, Any, Tuple, List

# canonical section header keywords (lowercase)
SECTION_HEADERS = [
    "summary",
    "objective",
    "skills",
    "technical skills",
    "education",
    "experience",
    "work experience",
    "projects",
    "certifications",
    "achievements",
    "awards",
    "publications",
    "open source",
    "languages",
    "competitions",
    "hobbies",
    "personal_info",
    "contact",
    "internship",
    "skills & technologies",
]

# Map alias to canonical simple key
CANONICAL_ALIAS_MAP = {
    "technical skills": "skills",
    "skillset": "skills",
    "work experience": "experience",
    "projects": "projects",
    "personal_info": "personal_info",
    "contact": "personal_info",
}

HEADER_LINE_RE = re.compile(r"^[\s\-•\u2022]*([A-Za-z0-9 &\-]{2,60}):?\s*$")
# This will match typical lines that look like "EDUCATION", "Projects:", "Skills -"


def _normalize_header_key(s: str) -> str:
    k = s.strip().lower()
    k = re.sub(r"[^a-z0-9\s]", " ", k)
    k = re.sub(r"\s+", " ", k).strip()
    if k in CANONICAL_ALIAS_MAP:
        return CANONICAL_ALIAS_MAP[k]
    if k in SECTION_HEADERS:
        # map to shorter canonical key by removing spaces
        return k.replace(" ", "_")
    # fallback to simplified key
    return k.replace(" ", "_")


def _is_header_line(line: str) -> Tuple[bool, str]:
    """
    Detect if line is a header. Returns (is_header, header_key)
    """
    if not line or len(line) < 2:
        return False, ""
    stripped = line.strip()
    # if it's ALL CAPS and short, treat as header
    words = stripped.split()
    if stripped.isupper() and len(words) <= 5 and len(stripped) < 60:
        return True, _normalize_header_key(stripped)
    # if line matches header regex and contains a known header keyword
    m = HEADER_LINE_RE.match(stripped)
    if m:
        candidate = m.group(1)
        cand_low = candidate.lower()
        for h in SECTION_HEADERS:
            if h in cand_low:
                return True, _normalize_header_key(h)
        # if it looks like a header even if not in SECTION_HEADERS, return it
        return True, _normalize_header_key(candidate)
    # also if line starts with known header word
    ln_low = stripped.lower()
    for h in SECTION_HEADERS:
        if ln_low.startswith(h):
            return True, _normalize_header_key(h)
    return False, ""


def split_into_sections_by_headers(text: str) -> Dict[str, str]:
    """
    Walk through lines, break when a header is seen, collect blocks.
    """
    lines = text.splitlines()
    sections = {}
    current_key = "preamble"
    sections[current_key] = []
    for ln in lines:
        is_hdr, hdr_key = _is_header_line(ln)
        if is_hdr:
            current_key = hdr_key or hdr_key or "misc"
            sections.setdefault(current_key, [])
            continue
        sections.setdefault(current_key, []).append(ln)
    # join blocks
    out = {}
    for k, block_lines in sections.items():
        # join non-empty lines
        filtered = [l for l in block_lines if l and l.strip()]
        out[k] = "\n".join(filtered).strip()
    return out


def run_sectioning_on_text(cleaned_text: str) -> Dict[str, Any]:
    """
    Public facade for sectioning.
    Returns dict: {"raw_sections": {...}, "section_confidence": {...}}
    Confidence is heuristic: 0.9 if header detected, 0.6 if inferred, else 0.2
    """
    if not cleaned_text:
        return {"raw_sections": {}, "section_confidence": {}}

    # naive split by double newline paragraphs first to improve header isolation
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned_text) if p.strip()]
    # Rebuild a text that preserves paragraph boundaries
    text_for_scan = "\n\n".join(paragraphs)

    raw_sections = split_into_sections_by_headers(text_for_scan)

    # assign simple confidences: header presence => high
    section_confidence = {}
    for k, v in raw_sections.items():
        conf = 0.2
        if k in (
            "skills",
            "education",
            "experience",
            "projects",
            "certifications",
            "summary",
            "languages",
            "achievements",
        ):
            conf = 0.9 if v else 0.0
        else:
            # if block length > 80 chars, medium confidence
            conf = 0.6 if len(v) > 80 else 0.2
        section_confidence[k] = float(conf)

    return {"raw_sections": raw_sections, "section_confidence": section_confidence}
