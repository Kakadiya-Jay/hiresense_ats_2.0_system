"""
Unit tests for scorer

Path: tests/unit/test_scorer.py

Run with: pytest -q
"""

import pytest
from src.phases.embeddings_matching.helpers.embeddings import load_sbert
from src.phases.scoring.helpers.scorer import score_candidate


@pytest.mark.fast
def test_scoring_sanity():
    jd = "Looking for a Python developer with experience in pandas, numpy and machine learning."
    candidate = {
        "skills": ["Python", "Pandas", "Scikit-Learn"],
        "projects": [
            {
                "title": "ML project",
                "summary": "Built a churn model using sklearn and pandas.",
            }
        ],
        "experience": ["2 years working as data engineer"],
    }
    model = load_sbert()
    out = score_candidate(candidate, jd, sbert_model=model)
    assert "final_score" in out
    assert 0.0 <= out["final_score"] <= 1.0
    assert "per_feature" in out
    assert isinstance(out["per_feature"], dict)


@pytest.mark.fast
def test_final_weight_clamping():
    # synthetic: test that final_weight respects clamp when alpha & beta large
    jd = "Data science role"
    candidate = {"skills": ["data"], "projects": ["proj"], "experience": ["exp"]}
    model = load_sbert()
    out = score_candidate(
        candidate,
        jd,
        sbert_model=model,
        alpha=1.0,
        beta=1.0,
        profile_base_weights={"skills": 0.9, "projects": 0.9, "experience": 0.9},
    )
    for f, v in out["per_feature"].items():
        assert 0.0 <= v["final_weight"] <= 1.0
        # contrib = sim * final_weight and sim in [0,1]
        assert 0.0 <= v["contrib"] <= 1.0
