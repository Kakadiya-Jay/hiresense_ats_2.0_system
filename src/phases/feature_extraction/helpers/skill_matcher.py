# src/phases/feature_extraction/helpers/skill_matcher.py
"""
Skill matcher with defensive coercion.

Accepts a candidate iterable (may contain non-strings) and normalizes them to strings.
Performs dictionary substring matches first, then a fuzzy match fallback via rapidfuzz.

Returns a list of match dicts:
  {"candidate": <original_candidate_str>, "matched": <canonical_or_None>, "score": float, "method": "dict"|"fuzzy"|"none"}
"""

from typing import List, Dict, Iterable
import os
import json
import logging

logger = logging.getLogger(__name__)

# rapidfuzz for fuzzy matching
from rapidfuzz import fuzz, process

# Load skill dictionary path from env or default resource path
SKILL_DICTIONARY_PATH = os.environ.get(
    "HIRESENSE_SKILL_DICT", "config/hiresense_skills_dictionary_v2.json"
)


def _safe_normalize_candidates(candidates: Iterable) -> List[str]:
    """
    Ensure every candidate is a non-empty string. Convert lists/dicts if present.
    """
    normalized = []
    for c in candidates or []:
        if c is None:
            continue
        if isinstance(c, str):
            s = c.strip()
            if s:
                normalized.append(s)
        else:
            # handle list/dict or other types by converting to string
            try:
                s = str(c).strip()
            except Exception:
                s = ""
            if s:
                normalized.append(s)
    # dedupe while preserving order
    seen = set()
    out = []
    for it in normalized:
        low = it.lower()
        if low not in seen:
            seen.add(low)
            out.append(it)
    return out


def _load_skill_dictionary(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            dj = json.load(fh)
            # support both list and dict formats
            if isinstance(dj, dict) and "skills" in dj:
                return dj["skills"]
            if isinstance(dj, list):
                return dj
    except FileNotFoundError:
        logger.warning(
            "Skill dictionary not found at %s, falling back to small builtin list", path
        )
    except Exception as e:
        logger.exception("Error loading skill dictionary: %s", e)
    # fallback minimal list
    return [
        "python",
        "java",
        "c++",
        "javascript",
        "react",
        "nodejs",
        "docker",
        "kubernetes",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "pandas",
        "sql",
        "nosql",
        "aws",
        "gcp",
    ]


# load canonical skills list at module import
CANONICAL_SKILLS = _load_skill_dictionary(SKILL_DICTIONARY_PATH)


def dict_and_fuzzy_match(candidates: Iterable, fuzzy_threshold: int = 85) -> List[Dict]:
    """
    candidates: iterable of candidate strings (may be list/dict etc).
    fuzzy_threshold: token sort ratio threshold (0-100)
    """
    safe_candidates = _safe_normalize_candidates(candidates)
    results = []
    canonical_lower = [s.lower() for s in CANONICAL_SKILLS]

    for c in safe_candidates:
        low = c.lower()
        matched = None
        score = 0.0
        method = "none"
        # exact substring match
        for i, canon in enumerate(canonical_lower):
            if canon in low or low in canon:
                matched = CANONICAL_SKILLS[i]
                score = 1.0
                method = "dict"
                break
        # fuzzy fallback
        if matched is None:
            try:
                match = process.extractOne(
                    c, CANONICAL_SKILLS, scorer=fuzz.token_sort_ratio
                )
            except Exception as e:
                logger.debug("rapidfuzz process.extractOne failure: %s", e)
                match = None
            if match:
                matched_candidate, match_score, _ = match
                if match_score >= fuzzy_threshold:
                    matched = matched_candidate
                    score = float(match_score) / 100.0
                    method = "fuzzy"

        results.append(
            {
                "candidate": c,
                "matched": matched,
                "score": float(score),
                "method": method,
            }
        )

    return results
