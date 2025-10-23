# src/api/routes/scorer.py
"""
FastAPI route for /score

Path: src/api/routes/score.py

Supports:
 - JSON POST: {"jd_text": "...", "candidate_json": {...}}
 - form-data multipart: resume_file (file) + jd_text (string)
The route will call your existing resume extractor when a file is uploaded.
"""

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import traceback

# imports - adjust these paths only if your extractor/service lives elsewhere
from src.phases.embeddings_matching.helpers.embeddings import load_sbert
from src.phases.scoring.helpers.scorer import score_candidate

# Replace this import with your actual extractor implementation path.
# The extractor should accept bytes and return the candidate_json dict used by the scorer.
try:
    from src.phases.text_extraction.helpers.extractor import (
        extract_candidate_json_from_file,
    )
except Exception:
    # Fallback stub if extractor isn't available at import time.
    def extract_candidate_json_from_file(file_bytes: bytes) -> Dict[str, Any]:
        # Minimal placeholder: return empty skeleton
        return {
            "skills": [],
            "projects": [],
            "experience": [],
            "education": [],
            "certifications": [],
        }


router = APIRouter()


@router.post("/score")
async def post_score(
    request: Request,
    jd_text: Optional[str] = Form(None),
    resume_file: Optional[UploadFile] = File(None),
):
    """
    POST /score
    - If multipart with resume_file: resume_file + jd_text (form fields)
    - Else can accept JSON body with keys jd_text and candidate_json

    Returns scoring JSON from score_candidate(...)
    """
    try:
        candidate_json = None

        if resume_file is not None:
            # Use extractor to build candidate_json from uploaded file bytes
            file_bytes = await resume_file.read()
            candidate_json = extract_candidate_json_from_file(file_bytes)
            # jd_text may be passed as form; ensure provided
            if not jd_text:
                raise HTTPException(
                    status_code=400,
                    detail="jd_text form field is required when uploading resume_file",
                )

        else:
            # Try parsing JSON body
            body = await request.json()
            jd_text = body.get("jd_text") if body else jd_text
            candidate_json = body.get("candidate_json") if body else None

        if not jd_text or not candidate_json:
            raise HTTPException(
                status_code=400,
                detail="Request must include jd_text and candidate_json (or resume_file + jd_text)",
            )

        # Load model lazily and call scorer
        sbert_model = load_sbert()
        result = score_candidate(candidate_json, jd_text, sbert_model=sbert_model)

        return JSONResponse(content=result)

    except HTTPException:
        # re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        # catch-all: return 500 with stack trace in detail for debugging (remove stack traces in production)
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}\n{tb}")
