# src/phases/feature_extraction/main.py
"""
Feature extraction main orchestrator.

Takes sectioned JSON (from sectioning phase) and extracts structured features
(skills, projects, experience, education, certifications, roles, etc.)
using spaCy, n-grams, simple rule-based parsing, SBERT embeddings, and fuzzy/dictionary matching.

This file is defensive about section formats (str, list, dict) by using
_coerce_to_text from phrase_extractor, and applies consistent block-splitting
using BLOCK_JOINER (default "\n\n").
"""

from typing import Dict, Any, List, Tuple
import copy
import uuid
import logging
import numpy as np

# helpers (ensure these helper modules exist under the helpers/ dir)
from .helpers.phrase_extractor import sentence_and_ngram_candidates, _coerce_to_text
from .helpers.skill_matcher import dict_and_fuzzy_match
from .helpers.date_parser import parse_date_string
from .helpers.embeddings import embed_texts, cosine_similarity

logger = logging.getLogger(__name__)

# Block joiner for coercing lists/dicts into readable text.
# Change to "\n- " if you want bullet-style joining later.
BLOCK_JOINER = "\n\n"

# Sample parent-category weights (0..1). Keep dynamicable later.
PARENT_CATEGORY_WEIGHTS = {
    "skills": 0.9,
    "projects": 0.95,
    "open_source": 0.95,
    "research_papers": 0.98,
    "experience": 0.85,
    "education": 0.6,
    "certifications": 0.6,
    "achievements": 0.5,
    "hobbies": 0.2,
    "personal_info": 0.1,
    "competitions": 0.5,
    "volunteering": 0.4,
    "extra_activity": 0.3,
    "social_accounts": 0.2,
    "roles": 0.85,
}


def _safe_text_for_section(raw_section_value: Any, joiner: str = BLOCK_JOINER) -> str:
    """
    Convert arbitrary section value (str|list|dict|None) to a normalized string.
    Uses _coerce_to_text but applies a consistent joiner for top-level lists/dicts.
    """
    coerced = _coerce_to_text(raw_section_value)
    # If coercion already joined with \n\n, optionally replace duplicates or apply joiner.
    if joiner != "\n\n":
        # convert double-newline blocks to desired joiner if necessary
        coerced = coerced.replace("\n\n", joiner)
    return coerced.strip()


def _collect_skill_candidates_from_sections(
    all_candidates: Dict[str, Dict],
) -> Tuple[List[str], Dict[str, str]]:
    """
    From the all_candidates mapping (category -> {"sentences":..., "candidates":..., "raw":...}),
    build a pool of candidates likely representing skills and map each candidate to its origin category.
    """
    skill_candidate_pool = []
    candidate_origin_map = {}
    # reasonable sections to consider for skills:
    skill_sections = {
        "skills",
        "projects",
        "experience",
        "open_source",
        "achievements",
        "certifications",
    }
    for cat, obj in all_candidates.items():
        if cat.lower() in skill_sections:
            for c in obj.get("candidates", []):
                # deduplicate before appending
                if c not in candidate_origin_map:
                    candidate_origin_map[c] = cat
                    skill_candidate_pool.append(c)
    return skill_candidate_pool, candidate_origin_map


def _map_skill_to_subcategory(canonical_name: str) -> str:
    """
    Heuristic mapping of a canonical skill name to a parent subcategory for simple grouping.
    Extend this mapping as you expand your skill dictionary.
    """
    low = canonical_name.lower() if canonical_name else ""
    if any(x in low for x in ["react", "angular", "vue", "html", "css", "javascript"]):
        return "web_frontend"
    if any(
        x in low
        for x in ["node", "express", "django", "flask", "spring", "asp.net", "backend"]
    ):
        return "web_backend"
    if any(x in low for x in ["aws", "azure", "gcp", "cloud"]):
        return "cloud"
    if any(x in low for x in ["pytorch", "tensorflow", "keras", "deep"]):
        return "deep_learning"
    if any(x in low for x in ["nlp", "transformer", "bert", "gpt", "llm"]):
        return "nlp"
    if any(x in low for x in ["sql", "mysql", "postgres", "mongodb", "redis"]):
        return "databases"
    return "programming_language"


def extract_features_from_sectioned(
    sectioned_json: Dict[str, Any], jd_text: str = None
) -> Dict[str, Any]:
    """
    Main orchestrator function.

    Args:
      sectioned_json: JSON produced by sectioning phase with at least:
          {
            "user_email": ...,
            "candidate_name": ...,
            "sections": { "skills": "...", "projects": [...], ... },
            "personal_info": {...}  # optional
          }
      jd_text: optional job description text for semantic matching

    Returns:
      feature_extracted_json following the Feature extraction output structure (skeleton).
    """
    if not isinstance(sectioned_json, dict):
        raise ValueError("sectioned_json must be a dict")

    sec = copy.deepcopy(sectioned_json)

    out = {
        "comment": "Feature extraction result",
        "user_email": sec.get("user_email"),
        "candidate_name": sec.get("candidate_name"),
        "features": {
            "skills": {},
            "roles": {},
            "experience": {},
            "education": {},
            "certifications": {},
            "hobbies": {},
            "personal_info": {},
            "projects": {},
            "competitions": {},
            "achievements": {},
            "open_source": {},
            "research_papers": {},
            "volunteering": {},
            "extra_activity": {},
            "social_accounts": {},
        },
        "meta": {
            "id": str(uuid.uuid4()),
            "source": sec.get("source", "resume_pdf"),
            "parent_weights": PARENT_CATEGORY_WEIGHTS,
        },
    }

    sections = sec.get("sections", {}) or {}
    all_candidates = {}

    # 1) Normalize each section and extract sentence & ngram candidates
    for cat, raw_value in sections.items():
        try:
            text = _safe_text_for_section(raw_value)
            if not text:
                continue
            sentences, candidates = sentence_and_ngram_candidates(text, max_n=3)
            all_candidates[cat] = {
                "sentences": sentences,
                "candidates": candidates,
                "raw": text,
            }
        except Exception as e:
            logger.exception("Failed to process section %s: %s", cat, e)
            # continue but keep a record
            all_candidates[cat] = {
                "sentences": [],
                "candidates": [],
                "raw": _safe_text_for_section(raw_value),
            }

    # 2) Skill matching using dictionary + fuzzy (rapidfuzz)
    skill_candidates, candidate_origin_map = _collect_skill_candidates_from_sections(
        all_candidates
    )
    try:
        skill_matches = dict_and_fuzzy_match(skill_candidates, fuzzy_threshold=85)
    except Exception as e:
        logger.exception("skill matching failed: %s", e)
        # fallback: empty matches
        skill_matches = []

    # Build skills output grouped by subcategory
    skills_out = {}
    for m in skill_matches:
        canonical = m.get("matched") or m.get("candidate")
        origin = candidate_origin_map.get(m.get("candidate"))
        subcat = _map_skill_to_subcategory(canonical)
        skills_out.setdefault(subcat, []).append(
            {
                "name": canonical,
                "evidence": m.get("candidate"),
                "score": float(m.get("score", 0.0)),
                "method": m.get("method", "none"),
                "origin_section": origin,
            }
        )
    out["features"]["skills"] = skills_out

    # 3) Projects & Experience extraction (simple heuristics)
    projects_out = []
    experience_out = []
    for catname in ("projects", "experience", "open_source"):
        raw_val = sections.get(catname, "")
        sec_text = _safe_text_for_section(raw_val)
        if not sec_text:
            continue

        # Try splitting into blocks: double newlines OR bullet markers
        # keep simple: split on double newline first, else split on single newline if long
        blocks = [b.strip() for b in sec_text.split("\n\n") if b.strip()]
        if not blocks:
            # fallback to single-line splitting
            blocks = [b.strip() for b in sec_text.split("\n") if b.strip()]

        for block in blocks:
            first_line = block.split("\n", 1)[0][:200]
            date_info = parse_date_string(block)
            sents, candidates = sentence_and_ngram_candidates(block, max_n=3)
            techs = []
            for cand in candidates:
                match = next(
                    (mm for mm in skill_matches if mm.get("candidate") == cand), None
                )
                if match and match.get("matched"):
                    techs.append(match.get("matched"))
            entry = {
                "title": first_line,
                "dates": date_info,
                "role": None,
                "tech_stack": list(sorted(set(techs))),
                "summary": sents[0] if sents else "",
                "team_size": None,
                "evidence": block[:1000],
            }
            if catname == "projects" or "project" in first_line.lower():
                projects_out.append(entry)
            else:
                experience_out.append(
                    {
                        "company_name": first_line,
                        "role": None,
                        "years_of_experience": None,
                        "dates": date_info,
                        "project_details": entry,
                        "skills": list(sorted(set(techs))),
                        "team_size": None,
                        "other": None,
                    }
                )

    out["features"]["projects"] = {
        "comment": "extracted projects",
        "projects": projects_out,
    }
    out["features"]["experience"] = {
        "comment": "extracted experience",
        "experience": experience_out,
    }

    # 4) Roles: explicit roles section or infer from skills
    roles_section = sections.get("roles", "")
    roles_coerced = _safe_text_for_section(roles_section)
    if roles_coerced:
        roled = [
            r.strip() for r in roles_coerced.replace("\n", ",").split(",") if r.strip()
        ]
    else:
        # inference rules
        sl = [
            s_item["name"].lower()
            for sub in skills_out.values()
            for s_item in sub
            if s_item.get("name")
        ]
        inferred = []
        allskills = " ".join(sl)
        if any(x in allskills for x in ["pytorch", "tensorflow", "keras", "deep"]):
            inferred.append("ml engineer")
        if any(x in allskills for x in ["react", "node", "javascript", "fullstack"]):
            inferred.append("full stack developer")
        if any(x in allskills for x in ["aws", "azure", "gcp", "cloud"]):
            inferred.append("cloud engineer")
        roled = inferred
    out["features"]["roles"] = {"roles": roled}

    # 5) Education, certifications (line-based simple parse)
    out["features"]["education"] = {"comment": "parsed education", "education": []}
    edu_text = _safe_text_for_section(sections.get("education", ""))
    if edu_text:
        lines = [l.strip() for l in edu_text.split("\n") if l.strip()]
        for ln in lines:
            degree_type = []
            if any(
                k in ln.lower() for k in ["bachelor", "b.sc", "btech", "b.e", "b.s"]
            ):
                degree_type = ["ug"]
            elif any(
                k in ln.lower() for k in ["master", "m.sc", "ms", "m.tech", "mba"]
            ):
                degree_type = ["pg"]
            out["features"]["education"]["education"].append(
                {
                    "graduation": [
                        {
                            "graduation_type": degree_type,
                            "degree_name": ln[:200],
                            "university_name": None,
                            "result_method": [],
                            "result": None,
                            "year": None,
                        }
                    ]
                }
            )

    out["features"]["certifications"] = {"comment": "certs", "certifications": []}
    cert_text = _safe_text_for_section(sections.get("certifications", ""))
    if cert_text:
        for ln in [l.strip() for l in cert_text.split("\n") if l.strip()]:
            out["features"]["certifications"]["certifications"].append(
                {
                    "certificate_title": ln,
                    "issuing_organization": None,
                    "issue_date": None,
                    "expiration_date": None,
                    "credential_id": None,
                    "credential_url": None,
                    "skills": [],
                    "other": None,
                }
            )

    # 6) personal_info passed through (keep original if present)
    out["features"]["personal_info"] = sec.get(
        "personal_info", {"email": sec.get("user_email")}
    )

    # 7) achievements, competitions, open_source, research_papers, volunteering etc.
    for cat in (
        "achievements",
        "competitions",
        "open_source",
        "research_papers",
        "volunteering",
        "extra_activity",
        "social_accounts",
    ):
        text_val = _safe_text_for_section(sections.get(cat, ""))
        out["features"].setdefault(cat, {"comment": f"parsed {cat}", cat: []})
        if text_val:
            blocks = [b.strip() for b in text_val.split("\n\n") if b.strip()]
            if not blocks:
                blocks = [b.strip() for b in text_val.split("\n") if b.strip()]
            for b in blocks:
                out["features"][cat][cat].append({"summary": b[:1000], "raw": b})

    # 8) If JD provided, run SBERT similarity between JD and project summaries
    if jd_text:
        try:
            project_summaries = [
                p.get("summary", "") for p in projects_out if p.get("summary")
            ]
            # embed jd + project_summaries in one batch
            texts_to_embed = (
                [jd_text] + project_summaries if project_summaries else [jd_text]
            )
            emb_all = embed_texts(texts_to_embed)
            jd_emb = emb_all[0]
            proj_embs = emb_all[1:] if len(emb_all) > 1 else np.array([])
            if proj_embs.size:
                sims = cosine_similarity(np.array(proj_embs), jd_emb)
                for i, p in enumerate(projects_out):
                    try:
                        p["jd_similarity"] = float(sims[i, 0])
                    except Exception:
                        p["jd_similarity"] = None
            # compute overall skill-to-jd similarity (optional)
            # Could also compute candidate-level embedding and compare
        except Exception as e:
            logger.warning("Embedding/JD similarity failed: %s", e)

    # meta: include summary counts & simple signals
    try:
        out["meta"]["num_sections"] = len(sections)
        out["meta"]["num_skills_detected"] = (
            sum(len(v) for v in out["features"]["skills"].values())
            if out["features"]["skills"]
            else 0
        )
        out["meta"]["num_projects"] = len(projects_out)
    except Exception:
        pass

    return out
