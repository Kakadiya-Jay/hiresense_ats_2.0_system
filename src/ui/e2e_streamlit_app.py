# ui/streamlit_app.py
import streamlit as st
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd

st.set_page_config(page_title="HireSense — E2E Explainability", layout="wide")

st.title("HireSense — E2E demo (explainability)")

examples_dir = Path("examples")
default_output = examples_dir / "e2e_output.json"

# ----------------------------
# Local scoring function (copy of scripts/e2e_demo.py compute_final_score)
# ----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


model = load_model()


def normalize_weights(wdict):
    s = sum(wdict.values()) or 1.0
    return {k: (v / s) for k, v in wdict.items()}


def embed_texts(texts):
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    return model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)


def compute_final_score(
    candidate_json, jd_text, profile_weights=None, alpha=0.3, beta=0.2
):
    default_profile = {
        "projects": 0.30,
        "open_source": 0.15,
        "skills": 0.25,
        "experience": 0.20,
        "education": 0.05,
        "research_papers": 0.05,
    }
    profile_weights = profile_weights or default_profile

    def gather_evidence(features, key):
        ev = []
        obj = (features or {}).get(key)
        if not obj:
            return ev
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    ev += [str(x) for x in v]
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    parts = []
                    for f in (
                        "title",
                        "summary",
                        "tech_stack",
                        "repo_link",
                        "outcome",
                        "role",
                        "company",
                    ):
                        val = item.get(f)
                        if isinstance(val, list):
                            parts.append(" ".join(val))
                        elif val:
                            parts.append(str(val))
                    ev.append(" | ".join([p for p in parts if p]))
                else:
                    ev.append(str(item))
        return [e for e in ev if e]

    jd_emb = model.encode([jd_text], convert_to_numpy=True)[0]

    canonical = {
        "projects": "projects, outcomes, achievements",
        "open_source": "open source contributions, pull requests, repositories",
        "skills": "programming languages, frameworks, skills",
        "experience": "industry experience, role, company work",
        "education": "degree, grade, university",
        "research_papers": "published research, doi, journal",
    }

    per_feature = {}
    for feat in profile_weights.keys():
        evidence_list = gather_evidence(candidate_json.get("features", {}), feat)
        if not evidence_list:
            semantic_sim = 0.0
            candidate_strength = 0.0
            evidence = []
        else:
            emb = embed_texts(evidence_list)
            if emb.shape[0] == 0:
                semantic_sim = 0.0
            else:
                sims = (emb @ jd_emb) / (
                    (np.linalg.norm(emb, axis=1) * np.linalg.norm(jd_emb)) + 1e-9
                )
                semantic_sim = float(np.max(sims))
            strengths = []
            feats_obj = candidate_json.get("features", {}).get(feat) or []
            if isinstance(feats_obj, list):
                for item in feats_obj:
                    if (
                        isinstance(item, dict)
                        and item.get("feature_strength") is not None
                    ):
                        try:
                            strengths.append(float(item["feature_strength"]))
                        except Exception:
                            pass
            candidate_strength = float(np.mean(strengths)) if strengths else 0.8
            evidence = evidence_list

        canon_emb = model.encode([canonical.get(feat, feat)], convert_to_numpy=True)[0]
        jd_importance = float(
            (jd_emb @ canon_emb)
            / ((np.linalg.norm(jd_emb) * np.linalg.norm(canon_emb)) + 1e-9)
        )

        per_feature[feat] = {
            "base_weight": float(profile_weights.get(feat, 0.0)),
            "jd_importance": jd_importance,
            "candidate_strength": candidate_strength,
            "semantic_similarity": semantic_sim,
            "evidence": evidence,
        }

    un_norm = {}
    for k, v in per_feature.items():
        un_norm[k] = (
            v["base_weight"]
            + alpha * v["jd_importance"]
            + beta * v["candidate_strength"]
        )
    norm = normalize_weights(un_norm)

    final_score = 0.0
    for k, v in per_feature.items():
        contrib = v["semantic_similarity"] * norm[k]
        per_feature[k].update({"final_weight": norm[k], "contribution": contrib})
        final_score += contrib

    return {"final_score": float(final_score), "per_feature": per_feature}


# ----------------------------
# UI - load or upload
# ----------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.header("Load / Upload")
    if default_output.exists():
        if st.button("Load examples/e2e_output.json"):
            data = json.loads(default_output.read_text())
        else:
            data = None
    else:
        st.info(
            "No examples/e2e_output.json found. Run scripts/e2e_demo.py first or upload JSON."
        )
        data = None

    uploaded = st.file_uploader("Or upload e2e_output.json", type=["json"])
    if uploaded:
        try:
            data = json.load(uploaded)
            st.success("Uploaded JSON loaded")
        except Exception as e:
            st.error("Invalid JSON uploaded: " + str(e))
            data = None

with col2:
    if not data:
        st.write("No candidate loaded yet. Load sample or upload JSON.")
        st.stop()

    candidate = data.get("candidate", {})
    score = data.get("score_result", {})
    st.subheader(f"Candidate: {candidate.get('candidate_id','N/A')}")
    st.markdown(f"**Precomputed Final score:** `{score.get('final_score', 0.0):.4f}`")

    # sliders to override base weights interactively
    st.sidebar.header("Profile base weights (override)")
    default_profile = {
        k: v.get("base_weight", 0.0) for k, v in score.get("per_feature", {}).items()
    }
    if not default_profile:
        default_profile = {
            "projects": 0.3,
            "open_source": 0.15,
            "skills": 0.25,
            "experience": 0.2,
            "education": 0.05,
            "research_papers": 0.05,
        }
    editable_profile = {}
    for k, v in default_profile.items():
        editable_profile[k] = st.sidebar.slider(k, 0.0, 1.0, float(v), step=0.01)

    alpha = st.sidebar.slider("alpha (JD importance)", 0.0, 1.0, 0.3, step=0.01)
    beta = st.sidebar.slider("beta (candidate_strength)", 0.0, 1.0, 0.2, step=0.01)

    # JD text (optional override)
    jd_text = st.text_area("Job Description (used for re-scoring)", height=120)
    if not jd_text.strip():
        # try to get JD text from loaded JSON if available
        jd_text = (
            score.get("_jd_text")
            or "Looking for a Software Developer with Python and NLP experience."
        )

    if st.button("Re-score with current profile & JD"):
        new_result = compute_final_score(
            candidate, jd_text, profile_weights=editable_profile, alpha=alpha, beta=beta
        )
    else:
        new_result = score

    st.metric("Final score", f"{new_result.get('final_score', 0.0):.4f}")

    # breakdown table
    rows = []
    for feat, info in new_result.get("per_feature", {}).items():
        rows.append(
            {
                "feature": feat,
                "final_weight": round(info.get("final_weight", 0.0), 4),
                "semantic_sim": round(info.get("semantic_similarity", 0.0), 4),
                "contribution": round(info.get("contribution", 0.0), 4),
            }
        )
    df = pd.DataFrame(rows).set_index("feature")
    st.dataframe(df)

    st.subheader("Per-feature contribution chart")
    st.bar_chart(df["contribution"])

    # evidence expanders
    st.subheader("Evidence (per feature)")
    for feat, info in new_result.get("per_feature", {}).items():
        with st.expander(
            f"{feat} — sim {info.get('semantic_similarity',0.0):.3f} | weight {info.get('final_weight',0.0):.3f}"
        ):
            ev = info.get("evidence", []) or []
            if not ev:
                st.write("_No evidence extracted for this feature._")
            else:
                for i, e in enumerate(ev[:30], start=1):
                    st.write(f"{i}. {e}")
