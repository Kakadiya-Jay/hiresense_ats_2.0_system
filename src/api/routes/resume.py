# hiresense/src/api/routes/resume.py
"""
FastAPI routes for resume processing.
 - POST /process_resume : upload PDF -> returns doc_id + processed summary
 - POST /match_resume   : supply doc_id + job_description -> returns matches
 - POST /score_resume   : supply doc_id + job_description (+ optional required_keywords) -> returns final score
 - POST /score_batch    : score by list of doc_ids (JSON)
 - POST /score_batch/upload : upload multiple PDFs + score them (multipart)
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Tuple

from src.api.services.resume_service import (
    save_upload_bytes,
    process_pdf,
    get_stored_doc,
    match_job_description,
    score_resume,
    score_batch_by_doc_ids,
    process_and_score_files_batch,
)

router = APIRouter(prefix="/resume", tags=["resume"])


class ProcessResponse(BaseModel):
    doc_id: str
    sections: dict
    num_sentences: int


@router.post("/process_resume", response_model=ProcessResponse)
async def process_resume(file: UploadFile = File(...)):
    # Enforce PDF-only
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")
    content = await file.read()
    saved_path = save_upload_bytes(content, filename_hint=file.filename)
    processed = process_pdf(saved_path, persist=True)
    # trim big sections for response
    sections_preview = {
        k: (v[:1000] + ("..." if len(v) > 1000 else ""))
        for k, v in processed["sections"].items()
    }
    return ProcessResponse(
        doc_id=processed["doc_id"],
        sections=sections_preview,
        num_sentences=len(processed["sentences"]),
    )


class MatchRequest(BaseModel):
    doc_id: str
    job_description: str
    top_k: Optional[int] = 5


@router.post("/match_resume")
def match_resume(req: MatchRequest):
    if not get_stored_doc(req.doc_id):
        raise HTTPException(status_code=404, detail="doc_id not found")
    try:
        res = match_job_description(req.doc_id, req.job_description, top_k=req.top_k)
        return JSONResponse(content=res)
    except KeyError:
        raise HTTPException(status_code=404, detail="doc_id not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ScoreRequest(BaseModel):
    doc_id: str
    job_description: str
    required_keywords: Optional[str] = None  # comma-separated
    top_k: Optional[int] = 5


@router.post("/score_resume")
def score_resume_route(req: ScoreRequest):
    if not get_stored_doc(req.doc_id):
        raise HTTPException(status_code=404, detail="doc_id not found")
    keywords = (
        [k.strip() for k in req.required_keywords.split(",")]
        if req.required_keywords
        else None
    )
    try:
        res = score_resume(
            req.doc_id, req.job_description, required_keywords=keywords, top_k=req.top_k
        )
        return JSONResponse(content=res)
    except KeyError:
        raise HTTPException(status_code=404, detail="doc_id not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# New: Batch scoring endpoints
# ---------------------------


class ScoreBatchRequest(BaseModel):
    doc_ids: List[str]
    job_description: str
    required_keywords: Optional[str] = None  # comma-separated
    top_k: Optional[int] = 5


@router.post("/score_batch")
def score_batch(req: ScoreBatchRequest):
    """
    Score multiple stored docs by doc_id (JSON body).
    Example body:
    {
      "doc_ids": ["uuid1","uuid2"],
      "job_description": "Looking for Python developer with NLP",
      "required_keywords": "python,nlp",
      "top_k": 5
    }
    """
    keywords = (
        [k.strip() for k in req.required_keywords.split(",")]
        if req.required_keywords
        else None
    )
    try:
        res = score_batch_by_doc_ids(
            req.doc_ids,
            req.job_description,
            required_keywords=keywords,
            top_k=req.top_k,
        )
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score_batch/upload")
async def score_batch_upload(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    required_keywords: Optional[str] = Form(None),
    top_k: Optional[int] = Form(5),
):
    """
    Upload multiple PDF files in one request and score them against the job description.
    - files: list of PDFs (form-field with same name)
    - job_description: form string
    - required_keywords: comma separated form string (optional)
    - top_k: int (optional)
    """
    # Validate filenames and read bytes
    saved_uploads: List[Tuple[bytes, str]] = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"Only .pdf files allowed: {f.filename}"
            )
        content = await f.read()
        saved_uploads.append((content, f.filename))
    keywords = (
        [k.strip() for k in required_keywords.split(",")] if required_keywords else None
    )
    try:
        res = process_and_score_files_batch(
            saved_uploads, job_description, required_keywords=keywords, top_k=top_k
        )
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
