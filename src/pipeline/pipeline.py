# src/pipeline/pipeline.py
"""
Simple orchestrator that for now calls only Phase 1 (text extraction).
Later phases (sectioning, features, embeddings, scoring) will be added here.
"""

from src.phases.text_extraction.main import run_text_extraction_pipeline


def run_pipeline_for_pdf(
    pdf_path: str, jd_text: str = "", profile: str = "developer", store: bool = False
):
    # Phase 1
    meta = run_text_extraction_pipeline(pdf_path)
    # placeholder for next phases: sectioning, features, scoring...
    return {
        "candidate": meta,
        "score": None,
        "breakdown": {},
    }
