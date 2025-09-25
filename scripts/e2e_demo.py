# scripts/e2e_demo.py
"""
End-to-end demo script for HireSense MVP.
Usage:
    python scripts/e2e_demo.py

Expectations:
 - examples/sample_resume_demo.pdf exists
 - examples/sample_jd.txt exists
 - writes examples/e2e_output.json
 - You may need to adapt pipeline import path if your function name differs.
"""

import json
import sys
from pathlib import Path
import numpy as np

# try local sentence-transformers; if not present, instruct user
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print(
        "Missing dependency: sentence-transformers. Install via `pip install sentence-transformers`"
    )
    raise

# try to import your pipeline runner; try common locations
run_pipeline_for_pdf = None
possible_names = [
    "src.pipeline.pipeline.run_pipeline_for_pdf",
    "src.pipeline.run_pipeline_for_pdf",
    "src.pipeline.pipeline.process_resume",
    "src.api.pipeline.run_pipeline_for_pdf",
    "pipeline.run_pipeline_for_pdf",
    "src.pipeline.process_resume",
]
for p in possible_names:
    try:
        mod_path, func_name = p.rsplit(".", 1)
        __import__(mod_path, fromlist=[func_name])
        run_pipeline_for_pdf = getattr(sys.modules[mod_path], func_name)
        print(f"Using pipeline function: {p}")
        break
    except Exception:
        run_pipeline_for_pdf = None

if run_pipeline_for_pdf is None:
    # fallback stub to let you still test scoring flow with a synthetic candidate JSON
    def run_pipeline_for_pdf(pdf_path):
        print(
            "WARNING: Did not find a pipeline function in common paths. Using synthetic demo candidate."
        )
        return {
            "candidate_id": "demo_001",
            "features": {
                "projects": [
                    {
                        "title": "Smart Resume Parser",
                        "summary": "Built resume parser",
                        "tech_stack": ["python", "nlp"],
                        "feature_strength": 0.85,
                    }
                ],
                "skills": {
                    "programming": ["python", "django", "sql"],
                    "ml": ["sklearn", "pandas"],
                },
                "experience": [
                    {
                        "company": "Acme",
                        "role": "ML Engineer",
                        "duration_months": 24,
                        "feature_strength": 0.9,
                    }
                ],
                "education": [
                    {"degree": "B.Tech", "institute": "XYZ Univ", "year": 2021}
                ],
                "open_source": [
                    {"repo": "github.com/demo/resume-parser", "feature_strength": 0.6}
                ],
                "research_papers": [],
            },
        }


# ----------------------------
# Configurable model / defaults
# ----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"  # good default for speed
model = SentenceTransformer(MODEL_NAME)


# ----------------------------
# Scoring utilities
# ----------------------------
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
    """
    Return: {'final_score': float, 'per_feature': {...}}
    - profile_weights: dict mapping feature->base_weight
    - alpha: importance weight for JD similarity
    - beta: importance weight for candidate_strength
    """
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

    # encode JD
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
# Main demo flow
# ----------------------------
def main():
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    pdf = examples_dir / "sample_resume_demo.pdf"
    jd_file = examples_dir / "sample_jd.txt"
    if not pdf.exists():
        print(
            "Missing examples/sample_resume_demo.pdf — create or copy a resume there to run the pipeline."
        )
        # still allow synthetic candidate via run_pipeline_for_pdf fallback
    if not jd_file.exists():
        # create a small sample JD
        jd_file.write_text(
            "Looking for a Software Developer with Python, NLP, experience building resume parsing or search systems."
        )
        print("Wrote examples/sample_jd.txt (sample)")

    # run pipeline
    try:
        candidate = run_pipeline_for_pdf(str(pdf))
        print("Pipeline returned candidate id:", candidate.get("candidate_id", "N/A"))
    except Exception as e:
        print("Pipeline invocation failed:", e)
        print("Using synthetic candidate for scoring.")
        candidate = run_pipeline_for_pdf(str(pdf))

    jd_txt = jd_file.read_text()
    result = compute_final_score(candidate, jd_txt)
    out = {"candidate": candidate, "score_result": result}
    out_file = examples_dir / "e2e_output.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_file}. Final score: {result['final_score']:.4f}")


if __name__ == "__main__":
    main()
