# src/phases/scoring/helpers/scorer.py
"""
Resilient scorer with automatic fallback to feature-extraction when input is empty.

Behavior:
- Accepts either simplified candidate_json OR extractor-style object (with 'features' or 'sections').
- If the input is empty (no items for features), it will try to regenerate features by calling the feature-extraction
  facade: run_feature_extraction_from_sections(...) OR extract_features_from_sectioned(...) if available.
- Adds a 'debug' field to the returned payload explaining fallback actions.

Note: This file assumes the extractor module exists at src.phases.feature_extraction.main with
      run_feature_extraction_from_sections(...) or extract_features_from_sectioned(...).
      The code will try imports and continue gracefully if they are not available.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import os
import logging
import re

logger = logging.getLogger(__name__)

# Try to import extractor run functions (defensive)
_try_extractor = {}
try:
    from src.phases.feature_extraction.main import run_feature_extraction_from_sections, extract_features_from_sectioned  # type: ignore

    _try_extractor["run_feature_extraction_from_sections"] = (
        run_feature_extraction_from_sections
    )
    _try_extractor["extract_features_from_sectioned"] = extract_features_from_sectioned
except Exception:
    # imports may not be present in some deployments; we'll detect at runtime
    _try_extractor["run_feature_extraction_from_sections"] = None
    _try_extractor["extract_features_from_sectioned"] = None

# Now import scorer internals (embeddings helpers). If your project uses different paths, adjust imports.
try:
    from src.phases.embeddings_matching.helpers.embeddings import (
        embed_texts,
        cosine_sim_matrix,
        load_sbert,
    )
except Exception:
    # Provide local fallbacks to avoid hard failure; these will be very small no-op implementations.
    def embed_texts(texts, model=None):
        # fallback: return zeros array
        import numpy as _np

        if not texts:
            return _np.zeros((0, 768), dtype=_np.float32)
        return _np.zeros((len(texts), 768), dtype=_np.float32)

    def cosine_sim_matrix(a, b):
        import numpy as _np

        # fallback zero similarity
        if getattr(a, "size", 0) == 0 or getattr(b, "size", 0) == 0:
            return _np.zeros((0, 0))
        return _np.zeros((a.shape[0], b.shape[0]))

    def load_sbert():
        class Dummy:
            def get_sentence_embedding_dimension(self):
                return 768

        return Dummy()


# default scalars
ALPHA_DEFAULT = float(os.getenv("SCORE_ALPHA", 0.4))
BETA_DEFAULT = float(os.getenv("SCORE_BETA", 0.3))
TOP_K_FOR_AGG = int(os.getenv("TOP_K_FOR_AGG", 3))


# ---------- utility helpers ----------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def _is_features_empty(candidate_obj: Dict[str, Any]) -> bool:
    """
    Detect if a candidate object (either simplified or extractor format) has no feature items.
    Returns True when every known feature list is empty or not present.
    """
    if not isinstance(candidate_obj, dict):
        return True
    # if extractor-style
    if "features" in candidate_obj and isinstance(candidate_obj["features"], dict):
        f = candidate_obj["features"]
        # consider top lists under features
        keys = [
            "skills",
            "projects",
            "experience",
            "education",
            "certifications",
            "open_source",
            "research_papers",
        ]
        for k in keys:
            val = f.get(k)
            if isinstance(val, dict):
                # nested lists under dict (like {"projects": {"projects":[...]}})
                # search recursively
                for kk, vv in val.items():
                    if isinstance(vv, list) and len(vv) > 0:
                        return False
            elif isinstance(val, list) and len(val) > 0:
                return False
        return True
    # simplified candidate json
    keys = [
        "skills",
        "projects",
        "experience",
        "education",
        "certifications",
        "open_source",
        "research",
    ]
    any_present = False
    for k in keys:
        v = candidate_obj.get(k)
        if isinstance(v, list) and len(v) > 0:
            return False
        if v is not None:
            any_present = True
    # If no keys found, treat as empty
    if not any_present:
        return True
    return True


def _features_to_candidate_json(feature_extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map extractor output (with 'features') into simplified candidate_json expected by scorer.
    This is the same mapping used previously — kept here to avoid circular imports.
    """
    cand: Dict[str, Any] = {}
    feat = (
        feature_extracted.get("features", {})
        if isinstance(feature_extracted, dict)
        else {}
    )

    # SKILLS
    skills: List[str] = []
    skills_dict = feat.get("skills", {}) or {}
    if isinstance(skills_dict, dict):
        for subcat, items in skills_dict.items():
            if not items:
                continue
            for it in items:
                if isinstance(it, dict):
                    name = it.get("name") or it.get("skill") or it.get("title") or ""
                else:
                    name = str(it)
                if name:
                    skills.append(name)
    else:
        for it in skills_dict or []:
            skills.append(it.get("name") if isinstance(it, dict) else str(it))
    cand["skills"] = skills

    # PROJECTS
    projects_list: List[Dict[str, str]] = []
    projects_block = feat.get("projects", {}) or {}
    possible_projects = (
        projects_block.get("projects") or projects_block.get("project_list") or []
    )
    if not possible_projects and isinstance(feat.get("projects"), list):
        possible_projects = feat.get("projects")
    for p in possible_projects or []:
        if isinstance(p, dict):
            title = (
                p.get("title")
                or p.get("name")
                or (p.get("summary")[:60] if p.get("summary") else "")
            )
            summary = (
                p.get("summary") or p.get("description") or p.get("evidence") or ""
            )
            projects_list.append({"title": title, "summary": summary})
        else:
            projects_list.append({"title": str(p)[:80], "summary": ""})
    cand["projects"] = projects_list

    # EXPERIENCE
    exp_list: List[str] = []
    exp_block = feat.get("experience", {}) or {}
    possible_exps = exp_block.get("experience") or exp_block.get("positions") or []
    if not possible_exps:
        possible_exps = feat.get("work_experience") or feat.get("roles") or []
    for e in possible_exps or []:
        if isinstance(e, dict):
            summary = (
                e.get("summary")
                or e.get("description")
                or e.get("role")
                or e.get("company")
                or str(e)
            )
            exp_list.append(summary)
        else:
            exp_list.append(str(e))
    cand["experience"] = exp_list

    # EDUCATION
    edu_list: List[str] = []
    edu_block = feat.get("education", {}) or {}
    possible_edu = edu_block.get("education") or feat.get("education") or []
    for e in possible_edu or []:
        if isinstance(e, dict):
            degree = e.get("degree") or e.get("degree_name") or e.get("name") or str(e)
            edu_list.append(degree)
        else:
            edu_list.append(str(e))
    cand["education"] = edu_list

    # CERTIFICATIONS
    certs: List[str] = []
    cert_block = feat.get("certifications", {}) or {}
    cert_items = cert_block.get("certifications") or feat.get("certifications") or []
    for c in cert_items or []:
        if isinstance(c, dict):
            certs.append(
                c.get("certificate_title") or c.get("title") or c.get("name") or str(c)
            )
        else:
            certs.append(str(c))
    cand["certifications"] = certs

    # OPEN SOURCE / RESEARCH
    os_list: List[str] = []
    os_block = feat.get("open_source", {}) or {}
    os_items = (
        os_block.get("open_source")
        or feat.get("open_source")
        or feat.get("projects", [])
    )
    for it in os_items or []:
        if isinstance(it, dict):
            os_list.append(
                it.get("summary") or it.get("title") or it.get("name") or str(it)
            )
        else:
            os_list.append(str(it))
    cand["open_source"] = os_list

    # research alias
    research_items = feat.get("research_papers") or feat.get("research") or []
    research_list = []
    for r in research_items or []:
        if isinstance(r, dict):
            research_list.append(r.get("title") or r.get("summary") or str(r))
        else:
            research_list.append(str(r))
    cand["research"] = research_list

    return cand


# ---------- scoring primitives (same approach as before) ----------
def clamp_div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_candidate_strength(
    candidate_json: Dict[str, Any], feature_name: str
) -> float:
    items = candidate_json.get(feature_name, [])
    if not items:
        return 0.0
    strengths = []
    for it in items:
        if isinstance(it, dict):
            for key in ("strength", "confidence", "score", "feature_strength"):
                v = it.get(key)
                if v is not None:
                    try:
                        strengths.append(clamp(float(v), 0.0, 1.0))
                        break
                    except Exception:
                        pass
            else:
                if it.get("repo_link") or it.get("link") or it.get("metrics"):
                    strengths.append(0.8)
                else:
                    strengths.append(0.5)
        else:
            strengths.append(0.5)
    return float(np.mean([clamp(x, 0.0, 1.0) for x in strengths])) if strengths else 0.0


def compute_jd_importance_tfidf(candidate_texts: List[str], jd_text: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not candidate_texts or not jd_text:
        return 0.0
    try:
        corpus = [jd_text] + candidate_texts
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(corpus)
        mats = vec.transform(corpus)
        jd_vec = mats[0]
        cand_vecs = mats[1:]
        sims = cosine_similarity(cand_vecs, jd_vec).flatten()
        if sims.size == 0:
            return 0.0
        return float(max(0.0, min(1.0, float(sims.mean()))))
    except Exception:
        jd_lower = jd_text.lower()
        overlap = sum(1 for t in candidate_texts if t and t.lower() in jd_lower)
        return float(min(1.0, overlap / max(1, len(candidate_texts))))

from typing import Tuple
def compute_sim(candidate_texts: List[str], jd_text: str, model) -> Tuple[float, float, int]:
    if not candidate_texts:
        return 0.0, 0.0, -1
    try:
        jd_emb = embed_texts([jd_text], model=model)[0]
        cand_embs = embed_texts(candidate_texts, model=model)
        sims = cosine_sim_matrix(cand_embs, jd_emb)  # (n_cand, 1)
        sims = np.array(sims).flatten()
        best_idx = int(np.argmax(sims))
        best = float(sims[best_idx])
        avg = float(np.mean(sims))
        return best, avg, best_idx
    except Exception:
        return 0.0, 0.0, -1


# ---------- main API ----------
def score_candidate(
    candidate_json: Dict[str, Any],
    jd_text: str,
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
    profile_base_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Main entry point.
    - candidate_json: either simplified candidate JSON OR a full stored doc that may contain 'features', 'sections', or 'full_text'.
    - jd_text: job description text
    """
    debug_msgs: List[str] = []

    # First: if input looks like extractor-style or has 'features', check if empty and try to regenerate features
    if isinstance(candidate_json, dict) and _is_features_empty(candidate_json):
        debug_msgs.append("Input appears to have empty/missing features.")
        # If sections present, try run_feature_extraction_from_sections
        try:
            if (
                isinstance(candidate_json, dict)
                and "sections" in candidate_json
                and _try_extractor.get("run_feature_extraction_from_sections")
            ):
                debug_msgs.append(
                    "Attempting to regenerate features from 'sections' using run_feature_extraction_from_sections."
                )
                try:
                    regenerated = _try_extractor[
                        "run_feature_extraction_from_sections"
                    ](candidate_json.get("sections", {}), jd_text=jd_text)
                    if regenerated and isinstance(regenerated, dict):
                        candidate_json = _features_to_candidate_json(regenerated)
                        debug_msgs.append(
                            "Regenerated candidate_json from sections; continuing scoring."
                        )
                except Exception as e:
                    debug_msgs.append(f"Regeneration from sections failed: {e}")
            # If full_text present, try extract_features_from_sectioned or run_feature_extraction_from_sections on a single-section payload
            elif isinstance(candidate_json, dict) and (
                "full_text" in candidate_json
                or "cleaned_text" in candidate_json
                or "text" in candidate_json
            ):
                debug_msgs.append(
                    "Attempting to regenerate features from 'full_text' using extractor."
                )
                text_blob = (
                    candidate_json.get("full_text")
                    or candidate_json.get("cleaned_text")
                    or candidate_json.get("text")
                )
                # try extractor functions
                if _try_extractor.get("extract_features_from_sectioned"):
                    try:
                        regenerated = (
                            _try_extractor["extract_features_from_sectioned"](
                                {"sections": {"full": text_blob}}, jd_text=jd_text
                            )
                            if isinstance(text_blob, dict)
                            else _try_extractor["extract_features_from_sectioned"](
                                {"sections": {"full": text_blob}}, jd_text=jd_text
                            )
                        )
                        if regenerated and isinstance(regenerated, dict):
                            candidate_json = _features_to_candidate_json(regenerated)
                            debug_msgs.append(
                                "Regenerated candidate_json from full_text via extract_features_from_sectioned."
                            )
                    except Exception as e:
                        debug_msgs.append(
                            f"extract_features_from_sectioned failed: {e}"
                        )
                elif _try_extractor.get("run_feature_extraction_from_sections"):
                    try:
                        regenerated = _try_extractor[
                            "run_feature_extraction_from_sections"
                        ]({"full": text_blob}, jd_text=jd_text)
                        if regenerated and isinstance(regenerated, dict):
                            candidate_json = _features_to_candidate_json(regenerated)
                            debug_msgs.append(
                                "Regenerated candidate_json from full_text via run_feature_extraction_from_sections."
                            )
                    except Exception as e:
                        debug_msgs.append(
                            f"run_feature_extraction_from_sections on full_text failed: {e}"
                        )
        except Exception as e:
            debug_msgs.append(f"Unexpected error during regeneration attempt: {e}")

    # If still empty after regeneration attempt, return early with helpful debug
    if _is_features_empty(candidate_json):
        debug_msgs.append(
            "No features available after regeneration attempts. Returning zero-score with debug."
        )
        # return zero-scoring response with debug
        empty_per_feature = {
            "skills": {
                "sim": 0.0,
                "best_sim": 0.0,
                "evidence_index": "",
                "base_weight": 0.35,
                "jd_importance": 0.0,
                "strength": 0.0,
                "final_weight": 0.35,
                "contrib": 0.0,
                "items_count": 0,
            },
            "projects": {
                "sim": 0.0,
                "best_sim": 0.0,
                "evidence_index": "",
                "base_weight": 0.25,
                "jd_importance": 0.0,
                "strength": 0.0,
                "final_weight": 0.25,
                "contrib": 0.0,
                "items_count": 0,
            },
            "experience": {
                "sim": 0.0,
                "best_sim": 0.0,
                "evidence_index": "",
                "base_weight": 0.2,
                "jd_importance": 0.0,
                "strength": 0.0,
                "final_weight": 0.2,
                "contrib": 0.0,
                "items_count": 0,
            },
            "open_source": {
                "sim": 0.0,
                "best_sim": 0.0,
                "evidence_index": "",
                "base_weight": 0.05,
                "jd_importance": 0.0,
                "strength": 0.0,
                "final_weight": 0.05,
                "contrib": 0.0,
                "items_count": 0,
            },
            "certifications": {
                "sim": 0.0,
                "best_sim": 0.0,
                "evidence_index": "",
                "base_weight": 0.05,
                "jd_importance": 0.0,
                "strength": 0.0,
                "final_weight": 0.05,
                "contrib": 0.0,
                "items_count": 0,
            },
            "education": {
                "sim": 0.0,
                "best_sim": 0.0,
                "evidence_index": "",
                "base_weight": 0.05,
                "jd_importance": 0.0,
                "strength": 0.0,
                "final_weight": 0.05,
                "contrib": 0.0,
                "items_count": 0,
            },
            "research": {
                "sim": 0.0,
                "best_sim": 0.0,
                "evidence_index": "",
                "base_weight": 0.05,
                "jd_importance": 0.0,
                "strength": 0.0,
                "final_weight": 0.05,
                "contrib": 0.0,
                "items_count": 0,
            },
        }
        return {
            "final_score": 0.0,
            "raw_score": 0.0,
            "denom": 1.0,
            "per_feature": empty_per_feature,
            "evidence": {k: "" for k in empty_per_feature.keys()},
            "alpha": float(ALPHA_DEFAULT),
            "beta": float(BETA_DEFAULT),
            "debug": debug_msgs,
        }

    # At this point, candidate_json should be a simplified mapping of lists
    # Ensure keys exist for expected features
    features_keys = [
        "skills",
        "projects",
        "experience",
        "education",
        "certifications",
        "open_source",
        "research",
    ]
    simplified = {k: candidate_json.get(k, []) for k in features_keys}

    # load model
    sbert_model = load_sbert()

    # prepare jd sentence embeddings
    jd_sents = (
        sentence_split(jd_text)
        if isinstance(jd_text, str) and jd_text.strip()
        else [jd_text] if jd_text else []
    )
    jd_embs = (
        embed_texts(jd_sents, model=sbert_model)
        if jd_sents
        else np.zeros(
            (0, sbert_model.get_sentence_embedding_dimension()), dtype=np.float32
        )
    )

    per_feature = {}
    raw_score = 0.0
    denom = 0.0

    # base weights (keep this default unless override passed)
    base_profile = {
        "skills": 0.35,
        "projects": 0.25,
        "experience": 0.20,
        "education": 0.05,
        "certifications": 0.05,
        "open_source": 0.05,
        "research": 0.05,
    }

    for f in features_keys:
        items = simplified.get(f, []) or []
        candidate_texts = []
        if f == "skills":
            for it in items:
                if isinstance(it, dict):
                    candidate_texts.append(it.get("name") or str(it))
                else:
                    candidate_texts.append(str(it))
        elif f == "projects":
            for it in items:
                if isinstance(it, dict):
                    candidate_texts.append(
                        it.get("summary") or it.get("title") or str(it)
                    )
                else:
                    candidate_texts.append(str(it))
        else:
            for it in items:
                if isinstance(it, dict):
                    candidate_texts.append(
                        it.get("summary")
                        or it.get("title")
                        or it.get("name")
                        or str(it)
                    )
                else:
                    candidate_texts.append(str(it))

        # compute sim primitives
        best_sim, avg_sim, best_idx = compute_sim(
            candidate_texts, jd_text, model=sbert_model
        )

        # compute strengths
        strength = compute_candidate_strength(simplified, f)

        # jd importance via TF-IDF
        jd_importance = (
            compute_jd_importance_tfidf(candidate_texts, jd_text)
            if candidate_texts
            else 0.0
        )

        base_w = float(base_profile.get(f, 0.0))
        final_w = clamp(base_w + alpha * jd_importance + beta * strength, 0.0, 1.0)
        contrib = float(final_w) * float(best_sim)

        per_feature[f] = {
            "sim": float(avg_sim),
            "best_sim": float(best_sim),
            "evidence_index": (
                int(best_idx) if best_idx is not None and best_idx != -1 else ""
            ),
            "base_weight": float(base_w),
            "jd_importance": float(jd_importance),
            "strength": float(strength),
            "final_weight": float(final_w),
            "contrib": float(contrib),
            "items_count": len(candidate_texts),
        }

        raw_score += contrib
        denom += final_w

    denom_safe = denom if denom > 0 else 1.0
    final_score = float(raw_score / denom_safe) if denom_safe > 0 else 0.0
    final_score = clamp(final_score, 0.0, 1.0)

    # build evidence mapping
    evidence = {}
    for f, v in per_feature.items():
        idx = v.get("evidence_index", "")
        items = simplified.get(f, [])
        try:
            if idx != "" and items:
                idx_int = int(idx)
                if 0 <= idx_int < len(items):
                    evidence[f] = items[idx_int]
                else:
                    evidence[f] = items[0] if items else ""
            else:
                evidence[f] = items[0] if items else ""
        except Exception:
            evidence[f] = items[0] if items else ""

    resp = {
        "final_score": float(final_score),
        "raw_score": float(raw_score),
        "denom": float(denom_safe),
        "per_feature": per_feature,
        "evidence": evidence,
        "alpha": float(alpha),
        "beta": float(beta),
        "debug": debug_msgs,
    }
    return resp
