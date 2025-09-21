# src/phases/sectioning/helpers/section_model.py
"""
Model-based sectioner wrapper.

- Looks for config file at config/sectioning_model.json or env var SECTIONING_MODEL_PATH.
- If model path present and transformers available, exposes `sectionize_with_model(text)` that returns same shape as rules.
- If not available, `is_model_available()` returns False and callers should use rules.
"""

import os
import json
from typing import List, Dict, Any

MODEL_CFG_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    "config",
    "sectioning_model.json",
)
ENV_MODEL_PATH = os.environ.get("SECTIONING_MODEL_PATH", None)

_model = None
_tokenizer = None
_pipeline = None
_labels = None


def load_model_if_available():
    global _model, _tokenizer, _pipeline, _labels
    model_path = None
    if ENV_MODEL_PATH:
        model_path = ENV_MODEL_PATH
    elif os.path.exists(MODEL_CFG_PATH):
        try:
            with open(MODEL_CFG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_path = cfg.get("model_path")
        except Exception:
            model_path = None
    if not model_path:
        return False
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            pipeline,
        )

        _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        _model = AutoModelForTokenClassification.from_pretrained(
            model_path, local_files_only=True
        )
        _pipeline = pipeline(
            "token-classification",
            model=_model,
            tokenizer=_tokenizer,
            aggregation_strategy="simple",
        )
        # labels can be read from model.config if present; fallback to default
        _labels = getattr(_model.config, "id2label", None)
        return True
    except Exception:
        # model or transformers not available
        _model = _tokenizer = _pipeline = None
        return False


# Attempt lazy load at import time (safe)
MODEL_AVAILABLE = load_model_if_available()


def is_model_available() -> bool:
    return MODEL_AVAILABLE and _pipeline is not None


def sectionize_with_model(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    If model available, run token-classification pipeline (or sequence tagger) and convert to sections.
    Implementation here is a conservative mapping:
     - split text into paragraphs/lines
     - send each line through model and assign predicted section label (e.g., 'B-EXPERIENCE', 'I-EXPERIENCE' or 'SKILLS')
     - group contiguous lines with same label into blocks and return normalized headers
    The expected labels must match your fine-tuned model's label set.
    """
    if not is_model_available():
        raise RuntimeError("Sectioning model not available")
    # Basic approach: run pipeline per line
    lines = text.splitlines()
    out_sections = {}
    current_label = None
    buffer_lines = []
    buffer_start = None
    for idx, line in enumerate(lines):
        if not line.strip():
            # flush existing buffer
            if buffer_lines and current_label:
                key = current_label.lower()
                text_block = "\n".join(buffer_lines).strip()
                out_sections.setdefault(key, []).append(
                    {
                        "header": current_label,
                        "normalized_header": key,
                        "text": text_block,
                        "start_offset": buffer_start,
                        "end_offset": buffer_start + len("\n".join(buffer_lines)),
                    }
                )
            buffer_lines = []
            current_label = None
            buffer_start = None
            continue
        try:
            preds = _pipeline(line)
            # pipeline returns list of {'word', 'entity_group'} if aggregation_strategy set
            if preds and isinstance(preds, list):
                label = preds[0].get("entity_group") or preds[0].get("entity")
            else:
                label = None
        except Exception:
            label = None
        if label:
            # treat label as header for the line classification
            if current_label is None:
                current_label = label
                buffer_lines = [line]
                # compute approximate start offset by searching text for this line
                buffer_start = text.find(line)
            elif label == current_label:
                buffer_lines.append(line)
            else:
                # flush old
                key = current_label.lower()
                text_block = "\n".join(buffer_lines).strip()
                out_sections.setdefault(key, []).append(
                    {
                        "header": current_label,
                        "normalized_header": key,
                        "text": text_block,
                        "start_offset": buffer_start,
                        "end_offset": buffer_start + len("\n".join(buffer_lines)),
                    }
                )
                # start new
                current_label = label
                buffer_lines = [line]
                buffer_start = text.find(line)
        else:
            # if model cannot decide, group under 'other' or continue buffer
            if current_label:
                buffer_lines.append(line)
            else:
                # start other
                current_label = "OTHER"
                buffer_lines = [line]
                buffer_start = text.find(line)
    # flush last
    if buffer_lines and current_label:
        key = current_label.lower()
        text_block = "\n".join(buffer_lines).strip()
        out_sections.setdefault(key, []).append(
            {
                "header": current_label,
                "normalized_header": key,
                "text": text_block,
                "start_offset": buffer_start,
                "end_offset": buffer_start + len("\n".join(buffer_lines)),
            }
        )
    return out_sections
