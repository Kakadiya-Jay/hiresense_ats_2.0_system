# src/phases/sectioning/main.py
"""
Public API for Phase 2 sectioning.

Functions:
- sectionize(meta) -> list of section dicts
- build_candidate_skeleton(sections, meta) -> candidate JSON skeleton (now includes research_papers stubs)
"""

from typing import Dict, List
import uuid

from .helpers.segmenter import (
    split_into_lines_with_page,
    detect_headings,
    build_sections_from_headings,
)


def sectionize(meta: Dict) -> List[Dict]:
    """
    meta: result from Phase 1 (must contain 'pages' list and optionally 'cleaned_text'/'full_text').
    Returns sections list (see helper docs).
    """
    pages = meta.get("pages", [])
    if not pages:
        # fallback: if only cleaned_text provided, treat as single page
        txt = meta.get("cleaned_text") or meta.get("full_text") or ""
        pages = [txt]
    lines = split_into_lines_with_page(pages)
    headings = detect_headings(lines, threshold=0.45)  # threshold can be tuned
    sections = build_sections_from_headings(lines, headings)
    return sections


def _map_links_to_sections(
    link_metadata: List[Dict], sections: List[Dict]
) -> List[Dict]:
    """
    For each link metadata entry, attempt to attach 'evidence' pointing to section_id and page.
    If page is available, use page to map to section whose start_page <= page <= end_page.
    Otherwise attempt to match by searching normalized_url text in section text (best-effort).
    """
    # build page -> section_id lookup ranges
    page_map = {}
    for sec in sections:
        sp = sec.get("start_page", 0)
        ep = sec.get("end_page", sp)
        for p in range(sp, ep + 1):
            page_map.setdefault(p, []).append(sec["section_id"])

    enriched = []
    for lm in link_metadata:
        entry = dict(lm)
        evidence = {
            "section_id": None,
            "page": entry.get("page"),
            "anchor_text": entry.get("anchor_text"),
        }
        page = entry.get("page")
        if page is not None and page in page_map:
            # pick first matching section id (document order)
            evidence["section_id"] = page_map[page][0]
        else:
            # best-effort: search for normalized_url in section text
            norm = entry.get("normalized_url") or ""
            found = False
            for sec in sections:
                if norm and norm in (sec.get("text") or ""):
                    evidence["section_id"] = sec["section_id"]
                    evidence["page"] = sec.get("start_page", evidence["page"])
                    found = True
                    break
            if not found:
                evidence["section_id"] = None
        entry["evidence"] = evidence
        enriched.append(entry)
    return enriched


def build_candidate_skeleton(sections: List[Dict], meta: Dict) -> Dict:
    """
    Create a basic candidate JSON skeleton pre-populated with easy mappings.
    Now also produces research_papers stubs based on link metadata and publications sections.
    """
    skeleton = {
        "personal_info": {},
        "skills": [],
        "projects": [],
        "education": [],
        "certifications": [],
        "publications": [],
        "research_papers": [],
        "publications_raw": [],
        "publications_links": [],
        "publications_section_ids": [],
        "publications_detected": False,
        "links": meta.get("link_metadata", []),
        "raw_sections": sections,
    }

    # If Phase1 already provided emails/phones, prefer them
    emails = meta.get("emails", []) or []
    phones = meta.get("phones", []) or []

    # minimal personal_info extraction (keeps previous heuristics)
    # ... reuse earlier heuristics (kept simple here)
    if sections:
        first_text = sections[0].get("text", "") or ""
        # find email/phone in first_text if not present
        import re

        email_re = re.compile(r"([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)")
        phone_re = re.compile(r"(?:\+\d{1,3}[-\s]?)?(?:\(\d{2,4}\)[-\s]?)?\d{5,12}")
        if not emails:
            m = email_re.search(first_text)
            if m:
                emails = [m.group(1).lower()]
        if not phones:
            m2 = phone_re.search(first_text)
            if m2:
                phones = [m2.group(0)]
        # naive name extraction: first non-empty line not containing email/phone
        for ln in first_text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if email_re.search(ln) or phone_re.search(ln):
                continue
            # pick the first plausible line as name
            skeleton["personal_info"]["name"] = ln
            break
    if emails:
        skeleton["personal_info"]["email"] = emails[0]
    if phones:
        skeleton["personal_info"]["phone"] = phones[0]

    # Enrich links: attach evidence mapping
    link_meta = meta.get("link_metadata", [])
    enriched_links = _map_links_to_sections(link_meta, sections)
    # replace skeleton links with enriched links
    skeleton["links"] = enriched_links

    # Build research_papers stubs:
    # 1) From enriched links where type == research_paper
    for lm in enriched_links:
        if lm.get("type") == "research_paper":
            stub = {
                "title": None,
                "authors": [],
                "year": None,
                "link": lm.get("normalized_url"),
                "type": lm.get("meta", {}).get("source", "research_paper"),
                "confidence": lm.get("confidence", 0.75),
                "evidence": lm.get("evidence"),
            }
            skeleton["research_papers"].append(stub)
            skeleton["publications_links"].append(lm.get("normalized_url"))
            if lm.get("evidence", {}).get("section_id"):
                skeleton["publications_section_ids"].append(
                    lm.get("evidence").get("section_id")
                )
                skeleton["publications_detected"] = True

    # 2) If there is a publications section, scan it for DOI/arXiv patterns and add stubs
    for sec in sections:
        if sec.get("section_type") == "publications":
            skeleton["publications_detected"] = True
            skeleton["publications_raw"].append(sec.get("text"))
            # quick regex scan inside section text
            text = sec.get("text") or ""
            # DOI
            import re

            doi_matches = re.findall(r"(10\.\d{4,9}/[^\s\)\]\,;]+)", text, flags=re.I)
            for d in doi_matches:
                norm = f"doi:{d.lower()}"
                stub = {
                    "title": None,
                    "authors": [],
                    "year": None,
                    "link": norm,
                    "type": "doi",
                    "confidence": 0.9,
                    "evidence": {
                        "section_id": sec.get("section_id"),
                        "page": sec.get("start_page"),
                    },
                }
                # avoid dups
                if not any(
                    s.get("link") == stub["link"] for s in skeleton["research_papers"]
                ):
                    skeleton["research_papers"].append(stub)
            # arXiv
            arxiv_matches = re.findall(
                r"([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", text, flags=re.I
            )
            for a in arxiv_matches:
                norm = f"https://arxiv.org/abs/{a}"
                stub = {
                    "title": None,
                    "authors": [],
                    "year": None,
                    "link": norm,
                    "type": "arxiv",
                    "confidence": 0.9,
                    "evidence": {
                        "section_id": sec.get("section_id"),
                        "page": sec.get("start_page"),
                    },
                }
                if not any(
                    s.get("link") == stub["link"] for s in skeleton["research_papers"]
                ):
                    skeleton["research_papers"].append(stub)

    # final candidate id & return
    skeleton["candidate_id"] = meta.get("candidate_id") or str(uuid.uuid4())
    return skeleton
