# src/phases/sectioning/helpers/section_rules.py
"""
Sectioning helpers for HireSense Phase-2.

Features:
- Robust header detection (line-start patterns + keyword fallback).
- Normalized matching (punctuation removal + lowercasing) for robust detection on messy resumes,
  while preserving offsets in the original cleaned_text.
- Project title-first grouping (each project block contains full project details).
- Certification splitting that groups certificate text with certificate links (from link_metadata).
- Other activities is final catch-all; phone numbers and URLs are removed from text and attached as metadata.
- Minimal, conservative heuristics; generalized for real-world resumes.

Primary functions:
- sectionize_text_by_rules(cleaned_text: str, link_metadata: list, phones: list) -> sections dict
- sectionize_from_extraction(extraction_json: dict) -> wrapper with candidate_name, sections, etc.

Notes:
- This file expects link_metadata items to have 'url' and 'start_offset' keys (if available).
- All offsets refer to positions in the original cleaned_text (not normalized).
"""

from typing import List, Dict, Any, Tuple
import re
import os
import json
import logging

logger = logging.getLogger(__name__)

# project root heuristic (adjust if necessary)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "project_triggers.json")

# Default header patterns (user can override via config/project_triggers.json)
DEFAULT_HEADERS = {
    "summary": [r"summary", r"objective", r"career objective"],
    "experience": [r"experience", r"work experience", r"professional experience", r"intern(?:ship|ships)?"],
    "projects": [r"projects?", r"personal projects"],
    "education": [r"education"],
    "skills": [r"skills", r"technical skills", r"programming languages"],
    "certifications": [r"certifications?", r"certificates?", r"certificate"],
    "awards": [r"awards", r"achievements?", r"recognition"],
    "other_activities": [r"activities", r"other activities", r"interests", r"hobbies", r"extracurricular"],
    "references": [r"references", r"referee"]
}

# common url/email/phone regexes
_URL_RE = re.compile(r"https?://[^\s\)\]\>]+|www\.[^\s\)\]\>]+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[\s\-\.]?)?(\d{3,4}[\s\-\.]?\d{3,4}[\s\-\.]?\d{0,4})")

# Month-year pattern helper
_MONTH_YEAR_RE = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}", flags=re.IGNORECASE)

# ---------- Utilities: normalized matching while preserving offsets ----------

def _load_section_header_overrides() -> Dict[str, List[str]]:
    """Load 'section_headers' overrides from config file if present."""
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("section_headers", {}) or {}
    except Exception as e:
        logger.warning("Could not load section headers override: %s", e)
        return {}

def _build_header_patterns() -> List[Tuple[str, re.Pattern]]:
    overrides = _load_section_header_overrides()
    merged = {}
    for k, v in DEFAULT_HEADERS.items():
        merged[k] = list(v)
    for k, v in overrides.items():
        if isinstance(v, list):
            merged[k] = list(dict.fromkeys([str(p) for p in v] + merged.get(k, [])))
    compiled = []
    for norm, pats in merged.items():
        for p in pats:
            # loose pattern: match the key as a word anywhere but prefer line-start matches later
            try:
                compiled.append((norm, re.compile(rf"(^|\n)\s*({p})\s*:?", flags=re.IGNORECASE)))
            except re.error:
                compiled.append((norm, re.compile(rf"(^|\n)\s*({re.escape(p)})\s*:?", flags=re.IGNORECASE)))
    return compiled

_COMPILED_PATTERNS = _build_header_patterns()

def _normalize_for_matching(text: str) -> str:
    """
    Return lowercased text with punctuation removed for robust pattern matching.
    We keep spaces so positions roughly correspond to words, but this normalized text
    is only used to detect tokens. Offsets used in output reference original text.
    """
    if text is None:
        return ""
    # replace many punctuation marks with a single space to preserve token boundaries
    normalized = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower().strip()

def _find_token_in_original(token: str, original: str, start_search: int = 0) -> int:
    """
    Find the first occurrence of token (case-insensitive) in original text starting from start_search.
    Returns index or -1. Token is expected to be a short substring (header token).
    """
    if not token:
        return -1
    try:
        # do case-insensitive search; escape token to treat as literal
        m = re.search(re.escape(token), original[start_search:], flags=re.IGNORECASE)
        if m:
            return start_search + m.start()
        return -1
    except Exception:
        # fallback to naive find lowercased
        low_orig = original.lower()
        pos = low_orig.find(token.lower(), start_search)
        return pos

# ---------- Primary header detection (pattern-based) ----------

def find_headers_positions_by_pattern_preserve_offsets(text: str) -> List[Dict[str, Any]]:
    """
    Use compiled patterns to find header tokens in the original text and return offsets
    referencing original text. Uses literal token matching to map normalized matches back.
    Returns list of dicts: [{'normalized': <key>, 'raw_header_line': <text>, 'start': int, 'end': int}, ...]
    """
    if not text:
        return []
    hits = []
    # Try line-start precision first on original text
    for norm, pat in _COMPILED_PATTERNS:
        for m in pat.finditer(text):
            start = m.start(2)
            end = m.end(2)
            raw = text[start:end]
            hits.append({"normalized": norm, "raw_header_line": raw, "start": start, "end": end})
    # sort + dedupe overlapped
    hits = sorted(hits, key=lambda x: x["start"])
    dedup = []
    last_end = -1
    for h in hits:
        if h["start"] <= last_end + 1:
            continue
        dedup.append(h)
        last_end = h["end"]
    return dedup

# ---------- Keyword-position fallback (works on normalized text) ----------

KEYWORD_LIST = [
    ("education", r"\beducation\b"),
    ("skills", r"\bskills\b|\btechnical skills\b"),
    ("projects", r"\bprojects?\b|\bpersonal projects\b"),
    ("experience", r"\bexperience\b|\bwork experience\b|\binternship\b"),
    ("certifications", r"\bcertifications?\b|\bcertificate\b"),
    ("awards", r"\bachievements?\b|\bawards\b"),
    ("other_activities", r"\bactivities\b|\binterests\b|\bhobbies\b")
]

def split_by_keyword_positions_with_mapping(original_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find keyword matches in normalized text, then map tokens back to positions in original text.
    Returns sections dict mapping normalized header -> list of blocks (with start/end offsets in original_text).
    """
    normalized = _normalize_for_matching(original_text)
    matches = []
    for norm, pat in KEYWORD_LIST:
        for m in re.finditer(pat, normalized, flags=re.IGNORECASE):
            token = m.group(0)
            matches.append({"normalized": norm, "token": token, "norm_start": m.start(), "norm_end": m.end()})
    if not matches:
        return {}

    # map tokens to original offsets: for each token, search from last mapped position for the literal token in original
    sections: Dict[str, List[Dict[str, Any]]] = {}
    last_search_pos = 0
    mapped = []
    for mt in matches:
        token = mt["token"]
        orig_pos = _find_token_in_original(token, original_text, start_search=last_search_pos)
        if orig_pos == -1:
            # try a global search
            orig_pos = _find_token_in_original(token, original_text, start_search=0)
        if orig_pos == -1:
            # skip if cannot map
            continue
        token_end = orig_pos + len(token)
        mapped.append({"normalized": mt["normalized"], "raw": original_text[orig_pos:token_end], "start": orig_pos, "end": token_end})
        last_search_pos = token_end + 1

    if not mapped:
        return {}

    # sort mapped and dedupe near duplicates
    mapped = sorted(mapped, key=lambda x: x["start"])
    dedup = []
    last = -1
    for m in mapped:
        if m["start"] <= last + 2:
            continue
        dedup.append(m)
        last = m["end"]
    mapped = dedup

    doc_len = len(original_text)
    for i, m in enumerate(mapped):
        content_start = m["end"]
        # prefer next newline after header if exists and not too far
        nl = original_text.find("\n", content_start)
        if nl != -1 and nl - content_start <= 120:
            content_start = nl + 1
        end_offset = mapped[i+1]["start"] if i+1 < len(mapped) else doc_len
        block_text = original_text[content_start:end_offset].strip()
        if not block_text:
            # expand content to include header itself
            block_text = original_text[m["start"]:end_offset].strip()
            content_start = m["start"]
        block = {
            "header": m["raw"],
            "normalized_header": m["normalized"],
            "text": block_text,
            "start_offset": content_start,
            "end_offset": end_offset
        }
        sections.setdefault(m["normalized"], []).append(block)
    return sections

# ---------- Project title-first grouping ----------

# patterns to detect probable project titles inside a block
_TITLE_DASH_RE = re.compile(r"(?m)(^|\n)\s{0,10}([A-Z][^\n]{2,120}?)\s*[-–—]\s+")
_TITLE_BRACKET_RE = re.compile(r"(?m)(^|\n)\s{0,10}([A-Z][^\n]{2,120}?)\s*\[[^\]]{2,120}\]")
_BULLET_TITLE_RE = re.compile(r"(?m)(^|\n)\s*[●\-\*]\s+([A-Z][^\n]{2,120}?)")
_MONTH_YEAR_RE = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}", flags=re.IGNORECASE)

def _find_project_title_positions(text: str) -> List[int]:
    """
    Return sorted unique positions likely to be the start of project titles.
    """
    positions = []
    for m in _TITLE_DASH_RE.finditer(text):
        positions.append(m.start(2))
    for m in _TITLE_BRACKET_RE.finditer(text):
        positions.append(m.start(2))
    for m in _BULLET_TITLE_RE.finditer(text):
        positions.append(m.start(2))
    # fallback signal: small chunk followed by bracket or month-year soon after
    if not positions:
        for m in re.finditer(r"(?m)(^|\n)\s{0,10}(.{5,80}?)\s*(?:\[|\(|\-|—|:|,)\s", text):
            start = m.start(2)
            window_text = text[m.end(2): m.end(2) + 160]
            if _MONTH_YEAR_RE.search(window_text):
                positions.append(start)
    positions = sorted(set(positions))
    return positions

def split_projects_title_first(project_text: str, block_start_offset: int) -> List[Dict[str, Any]]:
    """
    Split a single project block into per-project blocks using title-first grouping.
    Returns blocks with absolute offsets.
    """
    if not project_text or not project_text.strip():
        return []
    text = project_text
    doc_len = len(text)
    title_positions = _find_project_title_positions(text)
    if not title_positions:
        # fallback: attempt weak-split on bracket tokens or 'repo' or month-year
        # find bracketed tech or 'repo' occurrences to try splitting
        weak_positions = []
        for m in re.finditer(r"\[.*?\]|\brepo\b|" + _MONTH_YEAR_RE.pattern, text, flags=re.IGNORECASE):
            # backtrack to start of line
            ln_start = text.rfind("\n", 0, m.start())
            pos = ln_start + 1 if ln_start != -1 else max(0, m.start()-40)
            weak_positions.append(pos)
        title_positions = sorted(set(weak_positions))

    if not title_positions:
        # still none: return whole block
        return [{"text": text.strip(), "start_offset": block_start_offset, "end_offset": block_start_offset + len(text)}]

    # build slices
    blocks = []
    for i, pos in enumerate(title_positions):
        start = pos
        end = title_positions[i+1] if i+1 < len(title_positions) else doc_len
        chunk = text[start:end].strip()
        if not chunk:
            continue
        blocks.append({
            "text": chunk,
            "start_offset": block_start_offset + start,
            "end_offset": block_start_offset + end
        })
    # include any leading preamble by attaching to first block
    first_pos = title_positions[0]
    if first_pos > 0:
        pre = text[:first_pos].strip()
        if pre:
            blocks[0]["text"] = (pre + "\n" + blocks[0]["text"]).strip()
            blocks[0]["start_offset"] = block_start_offset
    return blocks

# ---------- Certification splitting & link association ----------

def _find_links_in_range(link_metadata: List[Dict[str, Any]], start: int, end: int) -> List[Dict[str, Any]]:
    """Return links whose start_offset falls within [start, end)."""
    if not link_metadata:
        return []
    hits = []
    for lm in link_metadata:
        ls = lm.get("start_offset", lm.get("start"))
        if ls is None:
            continue
        if start <= ls < end:
            hits.append(lm)
    return hits

def split_certificates_in_block(block: Dict[str, Any], link_metadata: List[Dict[str, Any]], cleaned_text: str) -> List[Dict[str, Any]]:
    """
    Split a single certification block text into multiple certificate blocks, associating nearby links.
    Returns list of certificate blocks with start/end offsets and 'links' field.
    """
    text = block.get("text", "") or ""
    s0 = block.get("start_offset", 0)
    e0 = block.get("end_offset", s0 + len(text))
    # identify candidate pieces by bullets/newlines or separators
    pieces = []
    if "●" in text or "\n" in text:
        raw_pieces = re.split(r"\n\s*[●\-\*]\s+|\n{2,}|\r\n{2,}", text)
        raw_pieces = [p.strip() for p in raw_pieces if p and len(p.strip()) > 2]
        pieces = raw_pieces
    if not pieces:
        # fallback splits on '|' or ';' or ' . '
        raw_pieces = re.split(r"\s*[/\|\;\•]\s*|\.\s+", text)
        raw_pieces = [p.strip() for p in raw_pieces if p and len(p.strip()) > 3]
        if raw_pieces:
            pieces = raw_pieces
    if not pieces:
        pieces = [text.strip()]

    # map piece offsets
    piece_entries = []
    cursor = s0
    for piece in pieces:
        # find piece in cleaned_text starting from cursor
        found = cleaned_text.find(piece, cursor, e0)
        if found == -1:
            found = cleaned_text.find(piece)
        if found == -1:
            p_start = cursor
        else:
            p_start = found
        p_end = p_start + len(piece)
        piece_entries.append({"text": piece, "start_offset": p_start, "end_offset": p_end, "links": []})
        cursor = p_end

    # attach links by nearest piece
    links = _find_links_in_range(link_metadata or [], s0, e0)
    for lm in links:
        ls = lm.get("start_offset") or lm.get("start")
        if ls is None:
            continue
        best_idx = None
        best_dist = None
        for idx, pe in enumerate(piece_entries):
            dist = abs(ls - pe["start_offset"])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            piece_entries[best_idx]["links"].append(lm)

    # build final blocks
    out_blocks = []
    for pe in piece_entries:
        combined_text = pe["text"]
        # append links text inline only if required; we keep link objects in 'links'
        out_blocks.append({
            "header": block.get("header", "certifications"),
            "normalized_header": "certifications",
            "text": combined_text.strip(),
            "start_offset": pe["start_offset"],
            "end_offset": pe["end_offset"],
            "links": pe.get("links", [])
        })
    return out_blocks

# ---------- Other activities cleanup (remove phones & urls from text) ----------

def _strip_urls_phones_from_text(text: str, links_in_block: List[Dict[str, Any]], phones_in_block: List[str]) -> str:
    # remove exact URL strings
    out = text
    for lm in links_in_block or []:
        url = lm.get("url") or lm.get("normalized_url") or ""
        if url:
            out = out.replace(url, " ")
            # try with trailing slash removed
            out = out.replace(url.rstrip("/"), " ")
    # remove phones heuristically
    for ph in phones_in_block or []:
        out = out.replace(ph, " ")
    # remove stray URLs (best-effort)
    out = re.sub(_URL_RE, " ", out)
    # collapse whitespace
    out = re.sub(r"\s+", " ", out).strip()
    return out

# ---------- Main sectionizer combining everything ----------

def sectionize_text_by_rules(cleaned_text: str,
                             link_metadata: List[Dict[str, Any]] = None,
                             phones: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Top-level rule-based sectionizer.

    Arguments:
      cleaned_text: the original cleaned text from Phase-1 (offsets refer to this string)
      link_metadata: list of link dicts, each preferably containing 'url' and 'start_offset'
      phones: list of phone strings with approximate offsets (if available as strings)

    Returns:
      sections mapping normalized_header -> list(blocks)
    """
    if not cleaned_text or not cleaned_text.strip():
        return {"other_activities": [{"header": None, "normalized_header": "other_activities", "text": "", "start_offset": 0, "end_offset": 0}]}

    # 1) try high-precision pattern-based detection on original text
    try:
        headers = find_headers_positions_by_pattern_preserve_offsets(cleaned_text)
    except Exception as e:
        logger.exception("Pattern header detection failed: %s", e)
        headers = []

    sections: Dict[str, List[Dict[str, Any]]] = {}
    doc_len = len(cleaned_text)

    if headers:
        for i, h in enumerate(headers):
            header_end = h["end"]
            nl = cleaned_text.find("\n", header_end)
            content_start = nl + 1 if nl != -1 else header_end
            end_offset = headers[i+1]["start"] if i+1 < len(headers) else doc_len
            block_text = cleaned_text[content_start:end_offset].strip()
            block = {
                "header": h["raw_header_line"].strip(),
                "normalized_header": h["normalized"],
                "text": block_text,
                "start_offset": content_start,
                "end_offset": end_offset
            }
            sections.setdefault(h["normalized"], []).append(block)
    else:
        # 2) fallback keyword-based detection with mapping to original offsets
        try:
            sections = split_by_keyword_positions_with_mapping(cleaned_text)
        except Exception as e:
            logger.exception("Keyword fallback detection failed: %s", e)
            sections = {}

    # 3) Projects: run title-first grouping on each projects block
    if "projects" in sections:
        new_projects = []
        for blk in sections["projects"]:
            splitted = split_projects_title_first(blk.get("text", ""), blk.get("start_offset", 0))
            # if splitted produced multiple pieces, map them to project blocks
            if splitted and len(splitted) > 1:
                for s in splitted:
                    new_projects.append({
                        "header": blk.get("header", "projects"),
                        "normalized_header": "projects",
                        "text": s["text"],
                        "start_offset": s["start_offset"],
                        "end_offset": s["end_offset"]
                    })
            else:
                # keep original as single project block
                new_projects.append(blk)
        sections["projects"] = new_projects

    # 4) Certifications: split each certification block into per-certificate blocks and attach links
    if "certifications" in sections:
        certs_acc = []
        for blk in sections["certifications"]:
            splitted = split_certificates_in_block(blk, link_metadata or [], cleaned_text)
            if splitted:
                certs_acc.extend(splitted)
            else:
                certs_acc.append(blk)
        sections["certifications"] = certs_acc

    # 5) Other activities: remove links & phones (they are metadata). Attach them separately as metadata fields.
    # If nothing else is present, we will later map entire document to other_activities
    if "other_activities" in sections:
        for blk in sections["other_activities"]:
            s = blk.get("start_offset", 0)
            e = blk.get("end_offset", s + len(blk.get("text", "")))
            # find links inside this block
            links_in_blk = _find_links_in_range(link_metadata or [], s, e)
            # find phones that appear in this text (phones param is list of strings)
            phones_in_blk = []
            if phones:
                for ph in phones:
                    if ph and cleaned_text.find(ph, s, e) != -1:
                        phones_in_blk.append(ph)
            # strip urls & phones from block text for cleanliness
            blk["links"] = links_in_blk
            blk["phones"] = phones_in_blk
            blk["text"] = _strip_urls_phones_from_text(blk.get("text", ""), links_in_blk, phones_in_blk)

    # 6) If no sections found, default the whole cleaned_text into other_activities
    if not sections:
        return {"other_activities": [{
            "header": None,
            "normalized_header": "other_activities",
            "text": _strip_urls_phones_from_text(cleaned_text, link_metadata or [], phones or []),
            "start_offset": 0,
            "end_offset": doc_len,
            "links": (link_metadata or []),
            "phones": (phones or [])
        }]}

    # 7) Final postprocess: remove tiny blocks, move them to other
    other_acc = []
    cleaned_sections: Dict[str, List[Dict[str, Any]]] = {}
    for k, blks in sections.items():
        kept = []
        for b in blks:
            txt = (b.get("text") or "").strip()
            b["text"] = txt
            if not txt or len(txt) < 6:
                other_acc.append(b)
            else:
                kept.append(b)
        if kept:
            cleaned_sections[k] = kept
    if other_acc:
        cleaned_sections.setdefault("other_activities", []).extend(other_acc)

    # 8) Ensure other_activities exists as catch-all: if any uncovered text remains between section blocks,
    # map it into other_activities (conservative). We can skip this for now to avoid overlapping coverage.
    return cleaned_sections

# ---------- Wrapper to accept extraction JSON and produce sectioned JSON ----------

def sectionize_from_extraction(extraction_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-level wrapper: accepts Phase-1 extraction JSON and returns Phase-2 sectioned JSON wrapper:
      {
        "candidate_name": ...,
        "emails": [...],
        "phones": [...],
        "link_metadata": [...],
        "sections": { ... },
        "source": "extraction",
        "meta": {...}
      }
    Behavior:
      - If candidate_name missing, this function does not attempt inference here (keep Phase-1/Phase-2 separation).
      - Phones passed through as list of strings if present.
    """
    cleaned = extraction_json.get("cleaned_text_updated") or extraction_json.get("raw_text") or ""
    link_metadata = extraction_json.get("link_metadata") or extraction_json.get("links") or []
    phones = []
    # try to get phones if stored as list of dicts or strings
    phs = extraction_json.get("phones") or extraction_json.get("phone_numbers") or []
    if isinstance(phs, list):
        # possible formats: ["+91 9...", {"phone": "...", "start_offset":..}, ...]
        for p in phs:
            if isinstance(p, str):
                phones.append(p)
            elif isinstance(p, dict):
                phones.append(p.get("phone") or p.get("value") or "")
    # call sectionizer
    sections = sectionize_text_by_rules(cleaned, link_metadata=link_metadata, phones=phones)
    # ensure other_activities exists if some text not assigned (policy: fallback to other_activities)
    if not sections:
        sections = {"other_activities": [{
            "header": None, "normalized_header": "other_activities",
            "text": _strip_urls_phones_from_text(cleaned, link_metadata, phones),
            "start_offset": 0, "end_offset": len(cleaned), "links": link_metadata, "phones": phones
        }]}
    out = {
        "candidate_name": extraction_json.get("candidate_name"),
        "emails": extraction_json.get("emails", []),
        "phones": phs,
        "link_metadata": link_metadata,
        "sections": sections,
        "source": "extraction",
        "meta": {
            "extraction_keys_present": list(extraction_json.keys())
        }
    }
    return out
