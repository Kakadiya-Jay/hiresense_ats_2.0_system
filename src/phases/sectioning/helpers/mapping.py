# src/phases/sectioning/helpers/mapping.py
"""
Map heading text to canonical section types.
Keep it small and configurable.
"""

from typing import Tuple
import re

MAPPING = {
    "education": ["education", "academic", "degree", "qualification", "qualification"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
    ],
    "projects": ["project", "projects"],
    "skills": ["skill", "skills", "technical skills", "technologies"],
    "certifications": ["certificate", "certification", "certificates"],
    "publications": [
        "publication",
        "publications",
        "research",
        "papers",
        "journal",
        "conference",
        "patent",
        "patents",
    ],
    "summary": ["summary", "objective", "profile"],
    "achievements": ["achievement", "achievements", "awards"],
    "contact": ["contact", "personal information", "personal info"],
    "interests": [
        "interest",
        "interests",
        "hobbies",
        "extracurricular",
        "volunteering",
    ],
}


def map_heading_to_type(heading: str) -> Tuple[str, float]:
    """
    Map a heading string to canonical section_type.
    Returns (section_type, confidence)
    """
    if not heading or not heading.strip():
        return ("unknown", 0.0)
    h = heading.lower()
    # exact keyword match
    for typ, aliases in MAPPING.items():
        for a in aliases:
            if re.search(r"\b" + re.escape(a) + r"\b", h):
                return (typ, 0.95)
    # fallback: heuristics based on words
    if "project" in h:
        return ("projects", 0.8)
    if "skill" in h or "technology" in h:
        return ("skills", 0.8)
    if "edu" in h or "degree" in h:
        return ("education", 0.8)
    if "publicat" in h or "research" in h or "paper" in h:
        return ("publications", 0.8)
    return ("unknown", 0.4)
