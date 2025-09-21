# src/phases/sectioning/main.py
"""
Phase 2 - Sectioning main.

- Exposes `sectionaize_extracted_text(extraction_json, use_model_fallback=True)`:
    Input: Phase-1 extraction JSON (must contain cleaned_text_updated or raw_text).
    Output: sectioned JSON wrapper:
      {
        "candidate_name": ...,
        "emails": [...],
        "phones": [...],
        "link_metadata": [...],
        "sections": { ... },   # normalized header -> list of blocks
        "source": "extraction",
        "meta": {...}
      }
- Integrates both rule-based and model-based helpers.
"""

from typing import Dict, Any
import logging
import os

from src.phases.sectioning.helpers.section_rules import sectionize_text_by_rules
from src.phases.sectioning.helpers.section_model import (
    is_model_available,
    sectionize_with_model,
)
from src.phases.sectioning.helpers.name_utils import infer_candidate_name

logger = logging.getLogger(__name__)


def _get_cleaned_text(extraction_json: Dict[str, Any]) -> str:
    return (
        extraction_json.get("cleaned_text_updated")
        or extraction_json.get("raw_text")
        or ""
    )


def sectionaize_extracted_text(
    extraction_json: Dict[str, Any], use_model_fallback: bool = True
) -> Dict[str, Any]:
    """
    Main function for phase-2 to be called by pipeline or API services.

    - If use_model_fallback is True and a model is available, model will be used and its output merged with rules.
    - Otherwise, rules-only path.
    """
    cleaned = _get_cleaned_text(extraction_json)
    if not cleaned:
        logger.warning("Empty cleaned text in extraction_json")
        sections = {
            "document": [
                {
                    "header": None,
                    "normalized_header": "document",
                    "text": "",
                    "start_offset": 0,
                    "end_offset": 0,
                }
            ]
        }
    else:
        # First apply rule-based sectioner (fast + precise)
        try:
            rule_sections = sectionize_text_by_rules(cleaned)
        except Exception as e:
            logger.exception("Rule-based sectioning failed: %s", e)
            rule_sections = {}

        model_sections = {}
        if use_model_fallback and is_model_available():
            try:
                model_sections = sectionize_with_model(cleaned)
            except Exception as e:
                logger.exception("Model-based sectioning failed: %s", e)
                model_sections = {}

        # Merge: give precedence to model for ambiguous or missing sections (configurable in future).
        merged = dict(rule_sections)  # shallow copy
        for k, v in model_sections.items():
            if k not in merged or not merged.get(k):
                merged[k] = v
            else:
                # if both exist, prefer rule-based but keep model variants in a 'model_{k}' optional bucket
                merged.setdefault(f"model_{k}", []).extend(v)

        # if merged empty fallback to document
        sections = (
            merged
            if merged
            else {
                "document": [
                    {
                        "header": None,
                        "normalized_header": "document",
                        "text": cleaned,
                        "start_offset": 0,
                        "end_offset": len(cleaned),
                    }
                ]
            }
        )

    # before creating 'out' wrapper, infer candidate name robustly:
    cand_name = infer_candidate_name(extraction_json)

    out = {
        "candidate_name": cand_name,
        "emails": extraction_json.get("emails", []),
        "phones": extraction_json.get("phones", []),
        "link_metadata": extraction_json.get("link_metadata", []),
        "sections": sections,
        "source": "extraction",
        "meta": {
            "used_model_fallback": bool(use_model_fallback and is_model_available()),
            "extraction_keys_present": list(extraction_json.keys()),
        },
    }
    return out


# If you want to allow simple CLI testing:
if __name__ == "__main__":
    import json, sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.phases.sectioning.main <extraction_json_file>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        ej = json.load(f)
    out = sectionaize_extracted_text(ej)
    print(json.dumps(out, indent=2, ensure_ascii=False))
