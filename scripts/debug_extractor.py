# scripts/debug_extractor.py
import pprint, sys
from src.api.services.resume_service import process_pdf
from src.phases.feature_extraction.main import extract_features_from_sectioned
from src.api.services.resume_service import _features_to_candidate_json

pdf = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "Jay_Kakadiya_Fresher_Resume_for_placement.pdf"
)
print("Processing PDF (no persist)...", pdf)
res = process_pdf(pdf, persist=False)
print("sections count:", len(res.get("sections", [])))
out = extract_features_from_sectioned(
    {"sections": res.get("sections"), "full_text": res.get("full_text")}
)
pp = pprint.PrettyPrinter(indent=2, width=140)
pp.pprint(
    {
        "num_skills_detected": out.get("meta", {}).get("num_skills_detected"),
        "skills_keys": list(out["features"]["skills"].keys()),
        "sample_skills": {
            k: [s.get("name") for s in v[:10]]
            for k, v in out["features"]["skills"].items()
        },
        "num_projects": out.get("meta", {}).get("num_projects"),
        "projects": out["features"].get("projects", {}).get("projects", [])[:3],
        "num_certs": len(
            out["features"].get("certifications", {}).get("certifications", [])
        ),
        "certs_sample": out["features"]
        .get("certifications", {})
        .get("certifications", [])[:5],
    }
)
cand = _features_to_candidate_json(out)
print("candidate_json preview keys:", list(cand.keys()))
pp.pprint(
    {
        k: cand.get(k)[:20] if isinstance(cand.get(k), list) else cand.get(k)
        for k in ("skills", "projects", "experience", "certifications")
    }
)
