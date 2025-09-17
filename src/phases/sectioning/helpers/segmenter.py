# src/phases/sectioning/helpers/segmenter.py
"""
Core segmentation utilities: split text into lines (with page info),
detect headings, and build sections.
"""

from typing import List, Dict, Tuple
from .header_rules import heading_score
from .mapping import map_heading_to_type


def split_into_lines_with_page(pages: List[str]) -> List[Dict]:
    """
    Given pages (list of page texts), return a list of lines with metadata:
    [ {"line": "...", "page": 0, "line_no": 0}, ... ]
    Keeps blank lines (text == "") to help split.
    """
    out = []
    for pidx, page in enumerate(pages):
        # split by newline but keep empty lines
        raw_lines = page.split("\n")
        for i, l in enumerate(raw_lines):
            out.append({"line": l.rstrip(), "page": pidx, "line_no": i})
    return out


def detect_headings(
    lines: List[Dict], threshold: float = 0.5
) -> List[Tuple[int, str, float]]:
    """
    Returns list of (index_in_lines, heading_text, score) for lines considered headings.
    The threshold defaults to 0.5 (tuneable).
    """
    candidates = []
    for idx, row in enumerate(lines):
        text = row.get("line", "")
        score = heading_score(text)
        if score >= threshold:
            candidates.append((idx, text.strip(), score))
    return candidates


def merge_small_sections(sections: List[Dict], min_lines: int = 2) -> List[Dict]:
    """
    If a section has very few lines (< min_lines), merge it into previous section
    (helps avoid fragments).
    """
    if not sections:
        return sections
    merged = [sections[0]]
    for sec in sections[1:]:
        if (
            len(merged[-1]["text"].splitlines()) < min_lines
            and len(sec["text"].splitlines()) < min_lines
        ):
            # merge to previous
            merged[-1]["text"] += "\n" + sec["text"]
            merged[-1]["end_page"] = sec["end_page"]
            merged[-1]["confidence"] = max(
                merged[-1].get("confidence", 0.5), sec.get("confidence", 0.5)
            )
        else:
            merged.append(sec)
    return merged


def build_sections_from_headings(
    lines: List[Dict], headings: List[Tuple[int, str, float]]
) -> List[Dict]:
    """
    headings: list of (idx, heading_text, score) already detected in order
    returns sections list with fields:
      { section_id, heading, section_type, text, start_page, end_page, confidence }
    """
    sections = []
    if not headings:
        # No headings detected -> entire document becomes one section
        full_text = "\n".join([r["line"] for r in lines])
        return [
            {
                "section_id": "s0",
                "heading": "",
                "section_type": "unknown",
                "text": full_text.strip(),
                "start_page": lines[0]["page"] if lines else 0,
                "end_page": lines[-1]["page"] if lines else 0,
                "confidence": 0.5,
            }
        ]

    for i, (idx, heading_text, score) in enumerate(headings):
        start_line = idx + 1
        end_line = (headings[i + 1][0] - 1) if i + 1 < len(headings) else len(lines) - 1
        block_lines = [lines[j]["line"] for j in range(start_line, end_line + 1)]
        block_text = "\n".join(block_lines).strip()
        # map heading to canonical type
        section_type, map_conf = map_heading_to_type(heading_text)
        # confidence combines heading score + mapping confidence
        confidence = min(1.0, 0.3 + score * 0.7 + map_conf * 0.2)
        sections.append(
            {
                "section_id": f"s{i}",
                "heading": heading_text,
                "section_type": section_type,
                "text": block_text,
                "start_page": (
                    lines[start_line]["page"]
                    if start_line < len(lines)
                    else lines[-1]["page"]
                ),
                "end_page": (
                    lines[end_line]["page"]
                    if end_line < len(lines)
                    else lines[-1]["page"]
                ),
                "confidence": round(confidence, 3),
            }
        )
    # merge tiny fragments
    sections = merge_small_sections(sections, min_lines=2)
    return sections
