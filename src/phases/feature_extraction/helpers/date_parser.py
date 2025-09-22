# src/phases/feature_extraction/helpers/date_parser.py
"""
Robust date parsing helper.

Accepts raw input as string, list, dict, or None. Uses phrase_extractor._coerce_to_text
to reliably convert to a single string before parsing. Returns a dict with:
  {"raw": <original_coerced_text>, "parsed": {...}, "confidence": float}
"""

import re
from typing import Dict, Any
import dateparser
import logging

logger = logging.getLogger(__name__)

# import coercion helper to normalize list/dict -> str
try:
    from .phrase_extractor import _coerce_to_text
except Exception:
    # fallback minimal coercion if import fails
    def _coerce_to_text(v):
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return "\n\n".join(str(x) for x in v)
        if isinstance(v, dict):
            return "\n\n".join(f"{k}: {v[k]}" for k in v)
        return str(v)


# Regex for month-range like "Jan 2020 - Mar 2021" including variations
MONTH_RANGE_RE = re.compile(
    r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{2,4})\s*[-–to]+\s*(Present|Now|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{2,4})",
    re.I,
)


def parse_date_string(raw_input: Any) -> Dict[str, Any]:
    """
    Parse a date string or block and return parsed start/end if found, plus confidence.
    Accepts raw_input of types: str | list | dict | None.
    """
    raw = _coerce_to_text(raw_input).strip()
    if not raw:
        return {"raw": raw, "parsed": {}, "confidence": 0.0}

    # Try explicit month-range regex first
    m = MONTH_RANGE_RE.search(raw)
    if m:
        start_raw, end_raw = m.group(1), m.group(2)
        try:
            start = dateparser.parse(start_raw)
        except Exception:
            start = None
        if end_raw and end_raw.lower() in ("present", "now"):
            end = None
        else:
            try:
                end = dateparser.parse(end_raw)
            except Exception:
                end = None
        parsed = {
            "start": start.strftime("%Y-%m") if start else None,
            "end": (end.strftime("%Y-%m") if end else "present"),
        }
        return {"raw": raw, "parsed": parsed, "confidence": 0.95}

    # Try to find any date-like token in the text (best-effort)
    try:
        dt = dateparser.parse(raw, settings={"PREFER_DAY_OF_MONTH": "first"})
        if dt:
            # treat as a single date (start)
            return {
                "raw": raw,
                "parsed": {"start": dt.strftime("%Y-%m")},
                "confidence": 0.7,
            }
    except Exception as e:
        logger.debug("dateparser.parse failed on input: %r -> %s", raw[:200], e)

    # fallback: try to detect year tokens (e.g., 2019, 2020-2021)
    year_match = re.search(r"(\b19|20)\d{2}\b", raw)
    if year_match:
        year = year_match.group(0)
        return {"raw": raw, "parsed": {"start": year}, "confidence": 0.4}

    # nothing found
    return {"raw": raw, "parsed": {}, "confidence": 0.1}
