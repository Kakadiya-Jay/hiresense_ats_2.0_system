# src/pipeline/skill_matcher.py
"""
Scoring module (SBERT + TF-IDF JD importance + normalized weights)
- Uses config/profile_config.json
- Returns a dict with per-feature contributions, evidence, final_score
"""
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

BASE = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE / "config"
PROFILE_CONF = CONFIG_DIR / "profile_config.json"

# ML imports
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# model
MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # fast and reasonably accurate


def load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


PROFILE_CFG = load_json(PROFILE_CONF)


# helpers
def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_jd_importance_tfidf(candidate_texts: List[str], jd_text: str) -> float:
    """
    Returns 0..1 representing the semantic/keyword importance of JD on candidate_texts using TF-IDF cosine.
    """
    if not candidate_texts:
        return 0.0
    corpus = [jd_text] + candidate_texts
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(corpus)
        mats = vec.transform(corpus)
        jd_vec = mats[0]
        cand_vecs = mats[1:]
        sims = cosine_similarity(cand_vecs, jd_vec).flatten()
        if sims.size == 0:
            return 0.0
        # mean similarity normalized: already between 0..1 for cosine
        return float(max(0.0, min(1.0, float(sims.mean()))))
    except Exception:
        # fallback: substring overlap
        jd_lower = jd_text.lower()
        overlap_count = 0
        for t in candidate_texts:
            if t and t.lower() in jd_lower:
                overlap_count += 1
        return float(min(1.0, overlap_count / max(1, len(candidate_texts))))


def compute_sim(candidate_texts: List[str], jd_text: str):
    """
    Returns (best_sim, avg_sim, best_idx)
    """
    if not candidate_texts:
        return 0.0, 0.0, -1
    try:
        jd_emb = MODEL.encode(jd_text, convert_to_tensor=True)
        cand_embs = MODEL.encode(candidate_texts, convert_to_tensor=True)
        sims = util.cos_sim(jd_emb, cand_embs).cpu().numpy().flatten()
        best_idx = int(np.argmax(sims))
        best = float(sims[best_idx])
        avg = float(np.mean(sims))
        return best, avg, best_idx
    except Exception:
        # fallback: no embeddings
        return 0.0, 0.0, -1


def normalize_profile_weights(profile_weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(profile_weights.values()) or 1.0
    return {k: float(v) / s for k, v in profile_weights.items()}


def score_candidate(
    candidate_json: Dict[str, Any],
    jd_text: str,
    profile_name: str = "developer",
    alpha: float = 0.4,
    beta: float = 0.3,
) -> Dict[str, Any]:
    """
    Main scoring API.
    candidate_json -> output of extractor
    jd_text -> job description
    profile_name -> keys present in config/profile_config.json
    alpha, beta -> jd_importance and candidate_strength multipliers
    """
    # load base profile weights (normalize to sum=1)
    profiles = PROFILE_CFG.get("profiles", {})
    base_profile = profiles.get(profile_name, profiles.get("developer", {}))
    base_profile_norm = normalize_profile_weights(base_profile)

    # features order and safe mapping from extractor output
    features_list = [
        "skills",
        "projects",
        "open_source",
        "experience",
        "education",
        "certifications",
        "research_papers",
    ]
    per_feature = {}
    evidence = {}
    raw_score = 0.0
    denom = 0.0

    for f in features_list:
        base_w = float(base_profile_norm.get(f, 0.05))

        # collect candidate_texts per feature
        candidate_texts = []
        if f == "skills":
            skills_dict = candidate_json.get("features", {}).get("skills", {})
            for cat, lst in (skills_dict or {}).items():
                if isinstance(lst, list):
                    for item in lst:
                        if isinstance(item, dict) and item.get("name"):
                            candidate_texts.append(item.get("name"))
                        else:
                            candidate_texts.append(item)
        elif f == "projects":
            projects = (
                candidate_json.get("features", {})
                .get("projects", {})
                .get("projects", [])
            )
            candidate_texts = [
                p.get("summary", "") or p.get("title", "") for p in projects
            ]
        elif f == "open_source":
            oss = (
                candidate_json.get("features", {})
                .get("open_source", {})
                .get("open_source", [])
            )
            candidate_texts = [
                o.get("summary", "") or o.get("repo_name", "") or o.get("repo_link", "")
                for o in oss
            ]
        elif f == "certifications":
            certs = (
                candidate_json.get("features", {})
                .get("certifications", {})
                .get("certifications", [])
            )
            candidate_texts = [c.get("certificate_title", "") for c in certs]
        elif f == "education":
            edus = (
                candidate_json.get("features", {})
                .get("education", {})
                .get("education", [])
            )
            candidate_texts = []
            for e in edus:
                if isinstance(e, dict):
                    # support nested structure: e may contain "graduation" list
                    if e.get("graduation"):
                        for g in e.get("graduation"):
                            candidate_texts.append(g.get("degree_name", "") or "")
                    else:
                        candidate_texts.append(e.get("degree_name", "") or "")
                else:
                    candidate_texts.append(str(e))
        elif f == "experience":
            exps = (
                candidate_json.get("features", {})
                .get("experience", {})
                .get("experience", [])
            )
            candidate_texts = [
                (
                    str(e.get("role", "") or "")
                    + " "
                    + str(e.get("company_name", "") or "")
                ).strip()
                for e in exps
                if isinstance(e, dict)
            ]
        elif f == "research_papers":
            rs = (
                candidate_json.get("features", {})
                .get("research_papers", {})
                .get("research_papers", [])
            )
            candidate_texts = [r.get("title", "") for r in rs]
        else:
            candidate_texts = []

        # semantic similarity primitives
        best_sim, avg_sim, best_idx = compute_sim(candidate_texts, jd_text)

        # candidate_strength_f: if feature has feature_strength entries use mean, else presence-based default
        strength = 0.0
        if f == "projects":
            projects = (
                candidate_json.get("features", {})
                .get("projects", {})
                .get("projects", [])
            )
            if projects:
                strengths = [p.get("feature_strength", 0.5) for p in projects]
                strength = float(np.mean(strengths))
        elif f == "open_source":
            oss = (
                candidate_json.get("features", {})
                .get("open_source", {})
                .get("open_source", [])
            )
            if oss:
                strengths = [o.get("feature_strength", 0.5) for o in oss]
                strength = float(np.mean(strengths))
        elif f == "experience":
            exps = (
                candidate_json.get("features", {})
                .get("experience", {})
                .get("experience", [])
            )
            if exps:
                strengths = [
                    e.get("feature_strength", 0.5)
                    for e in exps
                    if e.get("feature_strength", None) is not None
                ]
                strength = (
                    float(np.mean(strengths)) if strengths else (0.6 if exps else 0.0)
                )
        else:
            # for skills/certs/education, presence is signal
            strength = 0.75 if candidate_texts else 0.0

        # jd_importance using TF-IDF cosine
        jd_importance = (
            compute_jd_importance_tfidf(candidate_texts, jd_text)
            if candidate_texts
            else 0.0
        )

        # final_weight: use base + alpha*jd_importance*(1-base) + beta*strength*(1-base)
        residual = max(0.0, 1.0 - base_w)
        final_weight = (
            base_w + alpha * jd_importance * residual + beta * strength * residual
        )
        # cap final weight to 1.0
        final_weight = float(min(1.0, final_weight))

        contrib = float(best_sim) * final_weight

        per_feature[f] = {
            "sim": float(avg_sim),
            "best_sim": float(best_sim),
            "evidence_index": int(best_idx) if best_idx >= 0 else -1,
            "base_weight": float(base_w),
            "jd_importance": float(jd_importance),
            "strength": float(strength),
            "final_weight": float(final_weight),
            "contrib": float(contrib),
            "items_count": len(candidate_texts),
        }

        evidence[f] = (
            candidate_texts[best_idx]
            if (candidate_texts and 0 <= best_idx < len(candidate_texts))
            else ""
        )

        raw_score += contrib
        denom += final_weight

    final_score = safe_div(raw_score, denom) if denom > 0 else 0.0

    return {
        "final_score": float(final_score),
        "raw_score": float(raw_score),
        "denom": float(denom),
        "per_feature": per_feature,
        "evidence": evidence,
        "alpha": alpha,
        "beta": beta,
        "profile": profile_name,
        "doc_id": candidate_json.get("doc_id", ""),
    }
