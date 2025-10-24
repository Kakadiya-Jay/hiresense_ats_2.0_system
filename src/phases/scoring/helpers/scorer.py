# src/phases/scoring/helpers/scorer.py
"""
Scorer engine implementing:
  final_weight = clamp(base + alpha * jd_importance + beta * strength, 0, 1)
  contrib = final_weight * sim
  final_score = sum(contrib) / sum(final_weight)

This file is robust to two input shapes:
 - simplified candidate_json (top-level keys: 'skills','projects','experience',...)
 - extractor output shape: {"features": { ... }} (auto-converted)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import re
import os

# import embedder
from src.phases.embeddings_matching.helpers.embeddings import (
    embed_texts,
    cosine_sim_matrix,
    load_sbert,
)

# default hyperparameters (move to config if desired)
ALPHA_DEFAULT = float(os.getenv("SCORE_ALPHA", 0.4))  # updated default per request
BETA_DEFAULT = float(os.getenv("SCORE_BETA", 0.3))  # updated default per request
TOP_K_FOR_AGG = int(os.getenv("TOP_K_FOR_AGG", 3))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    # Very light-weight split; replace with spaCy if you prefer.
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def keyword_overlap_ratio(jd_text: str, feature_texts: List[str]) -> float:
    """
    Simple word-token overlap ratio between JD tokens and concatenated canonical feature texts.
    Returns ratio in [0,1].
    """
    if not jd_text or not feature_texts:
        return 0.0
    jd_tokens = set(re.findall(r"\w+", jd_text.lower()))
    feat_tokens = set()
    for t in feature_texts:
        feat_tokens.update(re.findall(r"\w+", str(t).lower()))
    if not jd_tokens:
        return 0.0
    overlap = jd_tokens.intersection(feat_tokens)
    return float(len(overlap)) / float(len(jd_tokens))


def compute_jd_importance(
    jd_text: str,
    feature_name: str,
    canonical_desc: Optional[List[str]],
    jd_embs: np.ndarray,
    feat_embs: np.ndarray,
    model=None,
) -> float:
    """
    Produce JD importance in [0,1] for a given feature:
      - keyword overlap vs canonical descriptors
      - semantic similarity (canonical_desc vs JD or fallback feat vs JD)
    Weigh keyword 0.4 and semantic 0.6 by default.
    """
    kw_ratio = 0.0
    sem_score = 0.0

    try:
        if canonical_desc:
            kw_ratio = keyword_overlap_ratio(jd_text, canonical_desc)
            # compute semantic similarity canonical_desc <-> jd_sents
            cd_embs = (
                embed_texts(canonical_desc, model=model)
                if len(canonical_desc) > 0
                else np.zeros((0, 0))
            )
            if cd_embs.size and jd_embs.size:
                sims = cosine_sim_matrix(cd_embs, jd_embs)  # (n_cd, n_jd)
                sem_score = float(np.max(sims)) if sims.size else 0.0
        else:
            # fallback: compute semantic similarity between feature items and JD
            if (
                feat_embs is not None
                and feat_embs.size
                and jd_embs is not None
                and jd_embs.size
            ):
                sims = cosine_sim_matrix(feat_embs, jd_embs)
                sem_score = float(np.max(sims)) if sims.size else 0.0
    except Exception:
        kw_ratio = kw_ratio if kw_ratio else 0.0
        sem_score = sem_score if sem_score else 0.0

    combined = 0.4 * clamp(kw_ratio, 0.0, 1.0) + 0.6 * clamp(sem_score, 0.0, 1.0)
    return clamp(combined, 0.0, 1.0)


def aggregate_feature_similarity(
    jd_embs: np.ndarray, feat_embs: np.ndarray, strategy: str = "topk_mean"
) -> Tuple[float, str, float]:
    """
    Aggregate similarity between JD sentence embeddings and feature item embeddings.
    Returns:
      - sim_val: aggregated similarity in [0,1]
      - evidence_idx: string index of best matching item (can be used to fetch evidence)
      - best_sim: raw best item->jd sim
    """
    if jd_embs is None or feat_embs is None or jd_embs.size == 0 or feat_embs.size == 0:
        return 0.0, "", 0.0

    sims = cosine_sim_matrix(feat_embs, jd_embs)  # (n_feat, n_jd)
    per_item_best = np.max(sims, axis=1)  # (n_feat,)

    if per_item_best.size == 0:
        return 0.0, "", 0.0

    if strategy == "max":
        best_idx = int(np.argmax(per_item_best))
        best_sim = float(per_item_best[best_idx])
        return clamp(best_sim, 0.0, 1.0), str(best_idx), best_sim

    elif strategy == "topk_mean":
        k = min(len(per_item_best), TOP_K_FOR_AGG)
        topk = np.sort(per_item_best)[-k:]
        mean_topk = float(np.mean(topk)) if topk.size else 0.0
        best_idx = int(np.argmax(per_item_best))
        best_sim = float(per_item_best[best_idx])
        return clamp(mean_topk, 0.0, 1.0), str(best_idx), best_sim

    else:  # "avg"
        mean_all = float(np.mean(per_item_best))
        best_idx = int(np.argmax(per_item_best))
        best_sim = float(per_item_best[best_idx])
        return clamp(mean_all, 0.0, 1.0), str(best_idx), best_sim


def compute_candidate_strength(
    candidate_json: Dict[str, Any], feature_name: str
) -> float:
    """
    Aggregate candidate-provided strength/confidence for a particular feature.
    Heuristic fallback if not provided.
    """
    items = candidate_json.get(feature_name, [])
    if not items:
        return 0.0

    strengths = []
    for it in items:
        if isinstance(it, dict):
            # prefer explicit numeric fields
            for key in ("strength", "confidence", "score", "feature_strength"):
                v = it.get(key)
                if v is not None:
                    try:
                        strengths.append(clamp(float(v), 0.0, 1.0))
                        break
                    except Exception:
                        pass
            else:
                # heuristic checks
                if it.get("repo_link") or it.get("link") or it.get("metrics"):
                    strengths.append(0.8)
                else:
                    strengths.append(0.5)
        else:
            # plain string item
            strengths.append(0.5)

    if strengths:
        return float(np.mean([clamp(x, 0.0, 1.0) for x in strengths]))
    return 0.0


# -------------------------
# Helper: convert extractor features -> simplified candidate_json
# -------------------------
def _features_to_candidate_json(feature_extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map extractor output (with 'features' nested) into simplified candidate_json expected by scorer.
    This internally mirrors resume_service._features_to_candidate_json to avoid circular imports.
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

    # research alias - map to 'research' key if present
    research_items = feat.get("research_papers") or feat.get("research") or []
    research_list = []
    for r in research_items or []:
        if isinstance(r, dict):
            research_list.append(r.get("title") or r.get("summary") or str(r))
        else:
            research_list.append(str(r))
    cand["research"] = research_list

    return cand


def score_candidate(
    candidate_json: Dict[str, Any],
    jd_text: str,
    profile_base_weights: Optional[Dict[str, float]] = None,
    canonical_feature_desc: Optional[Dict[str, List[str]]] = None,
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
    sbert_model=None,
) -> Dict[str, Any]:
    """
    Score candidate_json against jd_text and return an explainable score payload.

    candidate_json: dict with keys like 'skills', 'projects', etc. Each is a list of strings or dicts.
                    OR extractor-format with top-level key "features": {...}
    jd_text: job description text (string)
    profile_base_weights: optional mapping feature -> base weight
    canonical_feature_desc: optional mapping feature -> list[str] describing canonical descriptors (for jd importance)
    alpha, beta: tunable scalars
    sbert_model: a loaded SentenceTransformer instance (optional)
    """
    # Defensive: if caller passed extractor-style object containing 'features', convert it.
    if (
        isinstance(candidate_json, dict)
        and "features" in candidate_json
        and any(isinstance(candidate_json["features"], dict))
    ):
        candidate_json = _features_to_candidate_json(candidate_json)

    if not isinstance(candidate_json, dict):
        candidate_json = {}

    if profile_base_weights is None:
        profile_base_weights = {
            "skills": 0.35,
            "projects": 0.25,
            "experience": 0.20,
            "education": 0.05,
            "certifications": 0.05,
            "open_source": 0.05,
            "research": 0.05,
        }
    if canonical_feature_desc is None:
        canonical_feature_desc = {}

    if sbert_model is None:
        sbert_model = load_sbert()

    # Prepare JD sentence embeddings
    jd_sents = sentence_split(jd_text)
    if not jd_sents:
        # simple fallback: use full jd as single sentence
        jd_sents = [jd_text] if jd_text else []
    jd_embs = (
        embed_texts(jd_sents, model=sbert_model)
        if jd_sents
        else np.zeros(
            (0, sbert_model.get_sentence_embedding_dimension()), dtype=np.float32
        )
    )

    per_feature: Dict[str, Any] = {}
    total_raw = 0.0
    total_weights = 0.0

    # iterate union of candidate keys and default features
    features = set(list(candidate_json.keys()) + list(profile_base_weights.keys()))

    # embedding dimension for zero arrays
    dim = sbert_model.get_sentence_embedding_dimension()

    for f in features:
        items = candidate_json.get(f, [])
        feat_texts: List[str] = []
        for it in items:
            if isinstance(it, dict):
                t = (
                    it.get("summary")
                    or it.get("description")
                    or it.get("title")
                    or it.get("name")
                    or str(it)
                )
            else:
                t = str(it)
            if t:
                feat_texts.append(t)

        feat_embs = (
            embed_texts(feat_texts, model=sbert_model)
            if len(feat_texts) > 0
            else np.zeros((0, dim), dtype=np.float32)
        )

        # compute sim and evidence index
        sim_val, evidence_idx, best_sim = aggregate_feature_similarity(
            jd_embs, feat_embs, strategy="topk_mean"
        )

        # compute jd importance
        canonical_desc = canonical_feature_desc.get(f, [])
        jd_importance = compute_jd_importance(
            jd_text, f, canonical_desc, jd_embs, feat_embs, model=sbert_model
        )

        base_w = float(profile_base_weights.get(f, 0.0))
        strength = compute_candidate_strength(candidate_json, f)

        final_w = clamp(base_w + alpha * jd_importance + beta * strength, 0.0, 1.0)
        contrib = final_w * sim_val

        per_feature[f] = {
            "sim": float(sim_val),
            "best_sim": float(best_sim),
            "evidence_index": evidence_idx,
            "base_weight": float(base_w),
            "jd_importance": float(jd_importance),
            "strength": float(strength),
            "final_weight": float(final_w),
            "contrib": float(contrib),
            "items_count": len(feat_texts),
        }

        total_raw += contrib
        total_weights += final_w

    denom = total_weights if total_weights > 0 else 1.0
    final_score = float(total_raw / denom) if denom > 0 else 0.0
    final_score = clamp(final_score, 0.0, 1.0)

    # Build evidence mapping (map to original item where possible)
    evidence: Dict[str, Any] = {}
    for f, v in per_feature.items():
        idx = v.get("evidence_index", "")
        items = candidate_json.get(f, [])
        selected = None
        try:
            if idx != "" and items:
                idx_int = int(idx)
                if 0 <= idx_int < len(items):
                    selected = items[idx_int]
                else:
                    selected = items[0] if items else ""
            else:
                selected = items[0] if items else ""
        except Exception:
            selected = items[0] if items else ""
        evidence[f] = selected

    return {
        "final_score": final_score,
        "raw_score": float(total_raw),
        "denom": float(denom),
        "per_feature": per_feature,
        "evidence": evidence,
        "alpha": float(alpha),
        "beta": float(beta),
    }
