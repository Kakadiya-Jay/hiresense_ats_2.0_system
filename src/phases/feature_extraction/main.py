# src/phases/feature_extraction/main.py
"""
Feature extraction main orchestrator.

- Accepts sectioned JSON (from sectioning) via run_feature_extraction_from_sections
- Extracts structured features (skills, projects, experience, education, certifications, roles, etc.)
- Computes per-item `feature_strength` using config/feature_strength_rules.json
- Optionally computes JD similarity for project-level evidence when jd_text is passed.

Dependencies:
- .helpers.phrase_extractor -> sentence_and_ngram_candidates, _coerce_to_text
- .helpers.skill_matcher -> dict_and_fuzzy_match, _clean_candidate_text
- .helpers.date_parser -> parse_date_string
- .helpers.embeddings -> embed_texts, cosine_similarity
Ensure those helpers exist; this file provides the main orchestration.
"""

from typing import Dict, Any, List, Optional, Tuple
import copy
import uuid
import logging
import numpy as np
import re
import json
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# local helpers (assumed present in helpers/)
from .helpers.phrase_extractor import sentence_and_ngram_candidates, _coerce_to_text
from .helpers.skill_matcher import dict_and_fuzzy_match, _clean_candidate_text
from .helpers.date_parser import parse_date_string
from .helpers.embeddings import embed_texts, cosine_similarity

# configuration path
BASE = Path(__file__).resolve().parents[3]  # repo root
CONFIG_DIR = BASE / "config"

# Parent default weights (kept for reference in output)
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

BLOCK_JOINER = "\n\n"

def _load_skill_dict():
    p = "config/hiresense_skills_dictionary_v2.json"
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load hiresense_skills_dictionary_v2.json")
    # fallback minimal mapping
    return {"skills": {
        "programming": ["python", "java", "c++", "c", "c#", "javascript", "typescript"],
        "data": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
        "devops": ["docker", "kubernetes", "aws", "gcp", "azure"],
        "frontend": ["react", "angular", "vue", "html", "css", "javascript"]
    }}

_SKILL_DICT = _load_skill_dict()

def _map_skill_to_subcategory(skill_name: str) -> str:
    """
    Map a canonical skill string to a subcategory using the loaded skill dictionary.
    Returns a category key (string). If not found, returns 'misc'.
    Handles simple tokenization and fuzzy containment (case-insensitive).
    """
    if not skill_name:
        return "misc"
    try:
        s = str(skill_name).lower().strip()
        # exact match search in dictionary values
        skills_section = _SKILL_DICT.get("skills", {}) if isinstance(_SKILL_DICT, dict) else {}
        for cat, items in skills_section.items():
            if not items:
                continue
            for it in items:
                if not it:
                    continue
                it_s = str(it).lower().strip()
                # exact or token containment
                if s == it_s or s in it_s or it_s in s:
                    return cat
        # fallback: if skill contains common substrings
        if any(x in s for x in ["python", "pytorch", "tensorflow", "pandas", "numpy"]):
            return "data"
        if any(x in s for x in ["react", "angular", "vue", "html", "css", "frontend"]):
            return "frontend"
        if any(x in s for x in ["docker", "kubernetes", "aws", "gcp", "azure"]):
            return "devops"
    except Exception:
        logger.exception("Error mapping skill to subcategory for %s", skill_name)
    return "misc"

# -----------------------
# Utility helpers
# -----------------------
def _safe_text_for_section(raw_section_value: Any, joiner: str = BLOCK_JOINER) -> str:
    coerced = _coerce_to_text(raw_section_value)
    if joiner != BLOCK_JOINER:
        coerced = coerced.replace("\n\n", joiner)
    return coerced.strip()


def normalize_text_for_extraction(text: str, lowercase: bool = False) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[\u200B\u200C\u200D\uFEFF]", " ", t)
    t = re.sub(r"[\u2022\u2023\u25CF\u25AA\u25AB●•◦]", " ", t)
    t = "".join(ch if (ch == "\n" or ch == "\t" or ord(ch) >= 32) else " " for ch in t)
    t = re.sub(r"\n\s+\n", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    if lowercase:
        t = t.lower()
    return t


def _strip_leading_header(line: str) -> str:
    if not line:
        return ""
    l = line.strip()
    if re.match(r"^[A-Z0-9\-\s]{2,}:\s*$", l):
        return ""
    cleaned = re.sub(r"^[A-Z0-9\-\s]{2,}:\s*", "", l)
    return cleaned.strip()


def _normalize_cert_line(ln: str) -> str:
    if not ln or not ln.strip():
        return ""
    s = ln.strip().lstrip("•\u2022-•*· ").strip()
    s = re.sub(r"https?://\S+$", "", s).strip()
    s = re.sub(r"\(\s*https?://[^\)]+\)", "", s).strip()
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip(" ,;:.-")
    return s


# -----------------------
# Config readers
# -----------------------
def _load_config_json(fname: str) -> Dict[str, Any]:
    p = CONFIG_DIR / fname
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.exception("Failed to load config %s: %s", fname, e)
    return {}


FEATURE_RULES = _load_config_json("feature_strength_rules.json")
PROJECT_TRIGGERS = _load_config_json("project_triggers.json")


# -----------------------
# Feature-strength helpers
# -----------------------
def _compute_project_strength(
    summary: str, repo_link: Optional[str], tech_stack: List[str]
) -> float:
    """
    Compute a normalized project-level feature_strength using feature rules.
    """
    base = float(FEATURE_RULES.get("project_default_strength", 0.5))
    length_scale = float(FEATURE_RULES.get("project_length_score_scale", 150))
    link_boost = float(FEATURE_RULES.get("link_boost", 0.25))
    metric_boost_value = float(FEATURE_RULES.get("project_metric_boost", 0.9))

    len_score = min(1.0, len(summary) / max(1.0, length_scale))
    strength = base * 0.5 + len_score * 0.35
    if repo_link:
        strength = max(strength, min(1.0, strength + link_boost))
    if tech_stack:
        strength = max(strength, min(1.0, strength + 0.05 * len(tech_stack)))
    # detect metrics/outcomes and boost strongly
    if re.search(
        r"(improv|increase|reduce|decrease|accuracy|f1|roc|mse|latency|throughput|%|\d+\s*(hours|days|ms|s))",
        summary,
        re.I,
    ):
        strength = max(strength, metric_boost_value)
    return float(max(0.0, min(1.0, strength)))


def _compute_experience_strength(years: Optional[int]) -> float:
    """
    Map years of experience to a strength using feature rules.
    """
    if years is None:
        return float(FEATURE_RULES.get("experience_default_strength", 0.5))
    if years >= 5:
        return float(FEATURE_RULES.get("experience_years_gt_5", 0.95))
    if years >= 2:
        return float(FEATURE_RULES.get("experience_years_2_to_5", 0.8))
    return float(FEATURE_RULES.get("experience_years_lt_2", 0.55))


# -----------------------
# Main extraction function (existing robust implementation)
# -----------------------
def extract_features_from_sectioned(
    sectioned_json: Dict[str, Any], jd_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    This function performs feature extraction based on the pre-sectioned resume JSON.
    It is a defensive implementation that expects `sectioned_json` to be a dict with keys:
      - sections: {section_name: text_or_list_or_dict}
      - user_email, candidate_name, personal_info (optional)
    Returns structured JSON with features and per-item evidence.
    """
    if not isinstance(sectioned_json, dict):
        raise ValueError("sectioned_json must be a dict")

    sec = copy.deepcopy(sectioned_json)
    sections = sec.get("sections", {}) or {}

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

    all_candidates = {}

    # 1) Extract sentence & ngram candidates per section
    for cat, raw_value in sections.items():
        try:
            text_raw = _safe_text_for_section(raw_value)
            if not text_raw:
                continue
            text = normalize_text_for_extraction(text_raw, lowercase=False)
            sentences, candidates = sentence_and_ngram_candidates(text, max_n=3)
            all_candidates[cat] = {
                "sentences": sentences,
                "candidates": candidates,
                "raw": text_raw,
            }
        except Exception as e:
            logger.exception("Failed to process section %s: %s", cat, e)
            all_candidates[cat] = {
                "sentences": [],
                "candidates": [],
                "raw": _safe_text_for_section(raw_value),
            }

    # 2) Skill matching using dictionary + fuzzy
    skill_candidate_pool = []
    candidate_origin_map = {}
    for cat, obj in all_candidates.items():
        is_skill_section = False
        try:
            if any(
                k in cat.lower()
                for k in [
                    "skill",
                    "technical",
                    "technology",
                    "tools",
                    "framework",
                    "languages",
                    "skillset",
                ]
            ):
                is_skill_section = True
        except Exception:
            is_skill_section = False
        if is_skill_section or any(
            x in cat.lower()
            for x in ["project", "experience", "open source", "certific"]
        ):
            for c in obj.get("candidates", []) or []:
                if c:
                    if c not in candidate_origin_map:
                        candidate_origin_map[c] = cat
                        skill_candidate_pool.append(c)

    # fallback: aggregate top candidates
    if not skill_candidate_pool:
        for cat, obj in all_candidates.items():
            for c in obj.get("candidates", []) or []:
                if c and c not in candidate_origin_map:
                    candidate_origin_map[c] = cat
                    skill_candidate_pool.append(c)

    # clean + dedupe
    cleaned_candidates = []
    cleaned_origin_map = {}
    seen_clean = set()
    for c in skill_candidate_pool:
        cc = _clean_candidate_text(c)
        if not cc:
            continue
        k = cc.lower()
        if k in seen_clean:
            continue
        seen_clean.add(k)
        cleaned_candidates.append(cc)
        cleaned_origin_map[cc] = candidate_origin_map.get(c)

    skill_candidates = cleaned_candidates
    candidate_origin_map = cleaned_origin_map

    # fuzzy threshold tuned for recall (66..75 are typical). Keep 65 for higher recall but tune later.
    try:
        skill_matches = dict_and_fuzzy_match(skill_candidates, fuzzy_threshold=65.0)
    except Exception as e:
        logger.exception("skill matching failed: %s", e)
        skill_matches = [
            {"candidate": c, "matched": None, "score": 0.0, "method": "none"}
            for c in skill_candidates
        ]

    skills_out = {}
    for m in skill_matches:
        canonical = m.get("matched") or m.get("candidate")
        origin = (
            candidate_origin_map.get(m.get("candidate"))
            if isinstance(candidate_origin_map, dict)
            else None
        )
        subcat = _map_skill_to_subcategory(canonical) if canonical else "misc"
        canonical_str = canonical if isinstance(canonical, str) else str(canonical)
        skills_out.setdefault(subcat, []).append(
            {
                "name": canonical_str,
                "evidence": m.get("candidate"),
                "score": float(m.get("score", 0.0) or 0.0),
                "method": m.get("method", "none"),
                "origin_section": origin,
                "feature_strength": float(
                    0.95
                    if m.get("score", 0.0) >= 90
                    else (0.8 if m.get("score", 0.0) >= 70 else 0.6)
                ),
            }
        )
    out["features"]["skills"] = skills_out

    # 3) Projects & Experience extraction
    projects_out = []
    experience_out = []

    for catname in ("projects", "experience", "open_source"):
        raw_val = sections.get(catname, "")
        sec_text_raw = _safe_text_for_section(raw_val)
        if not sec_text_raw:
            continue
        sec_text = normalize_text_for_extraction(sec_text_raw, lowercase=False)
        raw_blocks = [
            b.strip() for b in re.split(r"\n{2,}|\r\n{2,}", sec_text) if b.strip()
        ]
        if not raw_blocks:
            raw_blocks = [b.strip() for b in sec_text.split("\n") if b.strip()]
        projects_seen = {}
        for block in raw_blocks:
            block_clean = re.sub(
                r"^\s*[A-Za-z \-]{2,30}:\s*", "", block, flags=re.I
            ).strip()
            if not block_clean:
                continue
            lines = [ln.strip() for ln in block_clean.splitlines() if ln.strip()]
            title = lines[0] if lines else ""
            title_clean = _clean_candidate_text(title) or title
            summary = (
                (" ".join(lines[1:4])[:1200]).strip()
                if len(lines) > 1
                else block_clean[:1200]
            )
            sents, candidates = sentence_and_ngram_candidates(block_clean, max_n=3)
            techs = []
            for cand in candidates or []:
                cand_clean = _clean_candidate_text(cand)
                matched = next(
                    (
                        mm
                        for mm in skill_matches
                        if mm.get("candidate")
                        and _clean_candidate_text(mm.get("candidate")) == cand_clean
                    ),
                    None,
                )
                if matched and matched.get("matched"):
                    techs.append(matched.get("matched"))
                elif isinstance(cand_clean, str) and re.search(
                    r"[A-Za-z\+#\.\-]{2,}", cand_clean
                ):
                    techs.append(cand_clean)
            key = title_clean.lower() if title_clean else summary[:60].lower()
            if key in projects_seen:
                ex = projects_seen[key]
                if len(summary) > len(ex["summary"]):
                    ex["summary"] = summary
                ex["evidence"] += "\n\n" + block[:500]
                continue
            # compute repo link if present
            replinks = re.findall(r"https?://\S+", block_clean)
            repo = next(
                (r for r in replinks if "github.com" in r or "gitlab.com" in r),
                (replinks[0] if replinks else ""),
            )
            entry = {
                "title": title_clean,
                "dates": parse_date_string(block_clean),
                "role": None,
                "tech_stack": list(sorted(set([t for t in techs if t]))),
                "summary": summary,
                "team_size": None,
                "evidence": block[:1000],
                "repo_link": repo or "",
            }
            # compute feature_strength for project
            entry["feature_strength"] = _compute_project_strength(
                entry["summary"], entry["repo_link"], entry["tech_stack"]
            )
            projects_seen[key] = entry
            if catname == "projects" or (title and "project" in title.lower()):
                projects_out.append(entry)
            else:
                # treat as experience block with embedded project
                years = None
                years_raw = re.search(r"(\d+)\s+years?", block_clean, re.I)
                if years_raw:
                    try:
                        years = int(years_raw.group(1))
                    except Exception:
                        years = None
                exp_entry = {
                    "company_name": title_clean,
                    "role": None,
                    "years_of_experience": years,
                    "dates": parse_date_string(block_clean),
                    "project_details": entry,
                    "skills": list(sorted(set([t for t in techs if t]))),
                    "team_size": None,
                    "other": None,
                    "feature_strength": _compute_experience_strength(years),
                }
                experience_out.append(exp_entry)

    out["features"]["projects"] = {
        "comment": "extracted projects",
        "projects": projects_out,
    }
    out["features"]["experience"] = {
        "comment": "extracted experience",
        "experience": experience_out,
    }

    # 4) Roles inference
    roles_section = sections.get("roles", "")
    roles_coerced = _safe_text_for_section(roles_section)
    if roles_coerced:
        roled = [
            r.strip() for r in roles_coerced.replace("\n", ",").split(",") if r.strip()
        ]
    else:
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
        if any(
            x in allskills
            for x in ["react", "node", "javascript", "fullstack", "full stack"]
        ):
            inferred.append("full stack developer")
        if any(x in allskills for x in ["aws", "azure", "gcp", "cloud"]):
            inferred.append("cloud engineer")
        roled = inferred
    out["features"]["roles"] = {"roles": roled}

    # 5) Education
    out["features"]["education"] = {"comment": "parsed education", "education": []}
    edu_text_raw = _safe_text_for_section(sections.get("education", ""))
    if edu_text_raw:
        edu_text = normalize_text_for_extraction(edu_text_raw)
        lines = [l.strip() for l in edu_text.split("\n") if l.strip()]
        for ln in lines:
            degree_type = []
            ln_low = ln.lower()
            if any(
                k in ln_low
                for k in ["bachelor", "b.sc", "btech", "b.e", "b.s", "bachelor's"]
            ):
                degree_type = ["ug"]
            elif any(
                k in ln_low
                for k in ["master", "m.sc", "ms", "m.tech", "mba", "master's"]
            ):
                degree_type = ["pg"]
            out["features"]["education"]["education"].append(
                {
                    "graduation": [
                        {
                            "graduation_type": degree_type,
                            "degree_name": _strip_leading_header(ln)[:200],
                            "university_name": None,
                            "result_method": [],
                            "result": None,
                            "year": None,
                        }
                    ]
                }
            )

    # 6) Certifications
    out["features"]["certifications"] = {"comment": "certs", "certifications": []}
    cert_text_raw = _safe_text_for_section(sections.get("certifications", ""))
    if cert_text_raw:
        cert_text = normalize_text_for_extraction(cert_text_raw)
        lines = [l.strip() for l in cert_text.split("\n") if l.strip()]
        cert_entries = []
        for ln in lines:
            cleaned = _normalize_cert_line(ln)
            if not cleaned:
                continue
            if re.match(r"^[A-Za-z\s]{2,20}:?$", cleaned) and len(cleaned.split()) <= 2:
                continue
            cert_entries.append(cleaned)
        seen = set()
        deduped = []
        for c in cert_entries:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        deduped = deduped[:20]
        for c in deduped:
            out["features"]["certifications"]["certifications"].append(
                {
                    "certificate_title": c,
                    "issuing_organization": None,
                    "issue_date": None,
                    "expiration_date": None,
                    "credential_id": None,
                    "credential_url": None,
                    "skills": [],
                    "other": None,
                    "feature_strength": 0.7,
                }
            )

    # 7) Achievements, competitions, open_source, research...
    for cat in (
        "achievements",
        "competitions",
        "open_source",
        "research_papers",
        "volunteering",
        "extra_activity",
        "social_accounts",
    ):
        text_val_raw = _safe_text_for_section(sections.get(cat, ""))
        out["features"].setdefault(cat, {"comment": f"parsed {cat}", cat: []})
        if text_val_raw:
            text_val = normalize_text_for_extraction(text_val_raw)
            blocks = [b.strip() for b in text_val.split("\n\n") if b.strip()]
            if not blocks:
                blocks = [b.strip() for b in text_val.split("\n") if b.strip()]
            for b in blocks:
                summary = _strip_leading_header(b) or b
                entry = {"summary": summary[:1000], "raw": b, "feature_strength": 0.4}
                # small boost for github links in open_source
                if cat == "open_source" and "github.com" in b.lower():
                    entry["feature_strength"] = float(
                        FEATURE_RULES.get("oss_default_strength", 0.6)
                    )
                # include in output list
                out["features"][cat][cat].append(entry)

    # 8) If JD provided, compute SBERT similarity for project summaries (adds jd_similarity field)
    if jd_text:
        try:
            project_summaries = [
                p.get("summary", "") for p in projects_out if p.get("summary")
            ]
            if project_summaries:
                texts_to_embed = [jd_text] + project_summaries
                emb_all = embed_texts(texts_to_embed)
                jd_emb = emb_all[0]
                proj_embs = emb_all[1:]
                if getattr(proj_embs, "shape", (0,))[0] > 0:
                    sims = cosine_similarity(np.array(proj_embs), jd_emb)
                    for i, p in enumerate(projects_out):
                        try:
                            p["jd_similarity"] = float(sims[i, 0])
                        except Exception:
                            p["jd_similarity"] = None
        except Exception as e:
            logger.warning("Embedding/JD similarity failed: %s", e)

    # meta summary counts
    try:
        out["meta"]["num_sections"] = len(sections)
        unique_skills = set()
        for sub in out["features"]["skills"].values():
            for s in sub:
                nm = s.get("name")
                if nm:
                    unique_skills.add(nm.lower())
        out["meta"]["num_skills_detected"] = len(unique_skills)
        out["meta"]["num_projects"] = len(projects_out)
    except Exception:
        pass

    return out


# Backwards-compatible facade expected by pipeline.py
def run_feature_extraction_from_sections(
    raw_sections: Dict[str, str],
    section_confidence: Optional[Dict[str, float]] = None,
    link_metadata: Optional[List[dict]] = None,
    emails: Optional[List[str]] = None,
    phones: Optional[List[str]] = None,
    jd_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Facade to run feature extraction given raw_sections and auxiliary metadata.
    Keeps signature light for APIs that pass sectioned payloads.
    """
    payload = {
        "sections": raw_sections or {},
        "user_email": emails[0] if emails else None,
        "candidate_name": None,
        "source": "resume_pdf",
    }
    # call the main extraction function and attach section confidences (if any)
    result = extract_features_from_sectioned(payload, jd_text=jd_text)
    if section_confidence:
        result.setdefault("meta", {})["section_confidence"] = section_confidence
    if link_metadata:
        result.setdefault("meta", {})["link_metadata"] = link_metadata
    return result
