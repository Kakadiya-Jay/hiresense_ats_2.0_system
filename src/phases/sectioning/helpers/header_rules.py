# src/phases/sectioning/helpers/header_rules.py
"""
Heading detection heuristics.
These are simple, explainable rules (no ML).
"""

import re
from typing import Tuple

# Common heading keywords and synonyms (lowercased)
HEADING_KEYWORDS = [
    "education",
    "experience",
    "work experience",
    "professional experience",
    "projects",
    "personal projects",
    "academic projects",
    "skills",
    "technical skills",
    "skills & technologies",
    "certificates",
    "certifications",
    "awards",
    "publications",
    "publication",
    "research",
    "papers",
    "journal",
    "conference",
    "patents",
    "research papers",
    "research publications",
    "publications & presentations",
    "summary",
    "objective",
    "profile",
    "contact",
    "personal information",
    "interests",
    "volunteering",
    "extracurricular",
    "hobbies",
]

# Precompile regex for quick checks
KEYWORD_RE = re.compile(
    r"\b(" + r"|".join([re.escape(k) for k in HEADING_KEYWORDS]) + r")\b", flags=re.I
)

# Short all-caps lines are often headings
ALL_CAPS_RE = re.compile(r"^[A-Z0-9 \-&]{2,80}$")

# Colon-terminated headings: "Experience:" or "Education :"
COLON_HEADING_RE = re.compile(r"^[A-Za-z0-9 \-&]{1,80}\s*:$")


def heading_score(line: str) -> float:
    """
    Returns a small score [0..1] estimating how likely 'line' is a heading.
    Uses multiple simple signals and sums them (capped at 1.0).
    """
    if not line or not line.strip():
        return 0.0
    s = 0.0
    text = line.strip()

    # Strong signal: keyword present as a whole word
    if KEYWORD_RE.search(text):
        s += 0.6

    # Strong signal: colon at end
    if COLON_HEADING_RE.match(text):
        s += 0.4

    # Medium signal: short (<=6 words) and title-case / uppercase-like
    words = text.split()
    if len(words) <= 6:
        # uppercase pattern
        if ALL_CAPS_RE.match(text):
            s += 0.5
        # title-case-ish (First word capitalized)
        elif text[0].isupper():
            s += 0.25

    # Medium signal: short & contains few punctuation chars
    if len(text) <= 60 and sum(1 for c in text if c in ",;()") <= 1:
        s += 0.05

    # Cap score at 1.0
    return min(1.0, s)
