"""
Scorer engine implementing:
  final_weight = clamp(base + alpha * jd_importance + beta * strength, 0, 1)
  contrib = final_weight * sim
  final_score = sum(contrib) / sum(final_weight)

Path: src/phases/scoring/helpers/scorer.py

Exposes:
 - score_candidate(candidate_json, jd_text, ...)
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
ALPHA_DEFAULT = float(os.getenv("SCORE_ALPHA", 0.3))
BETA_DEFAULT = float(os.getenv("SCORE_BETA", 0.2))
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
            for key in ("strength", "confidence", "score"):
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
    jd_text: job description text (string)
    profile_base_weights: optional mapping feature -> base weight
    canonical_feature_desc: optional mapping feature -> list[str] describing canonical descriptors (for jd importance)
    alpha, beta: tunable scalars
    sbert_model: a loaded SentenceTransformer instance (optional)
    """
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
    jd_embs = embed_texts(jd_sents, model=sbert_model)

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
