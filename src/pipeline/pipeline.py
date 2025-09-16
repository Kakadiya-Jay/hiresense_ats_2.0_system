# src/pipeline/pipeline.py
"""
Simple orchestrator stub for Day-0.
Replace stubs with real phase calls later.
"""
def run_pipeline_for_pdf(pdf_path: str, jd_text: str = "", profile: str = "developer", store: bool = False):
    # Minimal fake response to make API + UI work end-to-end for demos
    result = {
        "candidate_id": "demo-0001",
        "score": 0.0,
        "candidate": {
            "personal_info": {"name": "Demo Candidate", "email": None},
            "features": {},
            "link_metadata": [],
        },
        "explain": {"notes": "This is a pipeline stub. Replace with real pipeline steps."}
    }
    return result
