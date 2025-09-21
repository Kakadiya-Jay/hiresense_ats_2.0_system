# src/api/routes/resume.py
"""
FastAPI route for resume processing.
Endpoint: POST /process_resume

Accepts:
 - JSON body with "extraction_json": already-run Phase-1 extraction JSON, OR
 - multipart form upload with "file" (PDF) to run phase-1 extraction and then sectioning.

Returns:
 - sectioned JSON (Phase-2 output)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from src.api.services.resume_service import process_resume_payload

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/process_resume")
async def process_resume_route(
    extraction_json: Optional[dict] = Body(None),
    file: Optional[UploadFile] = File(None),
    use_model_fallback: bool = True,
):
    # Validation
    if extraction_json is None and file is None:
        raise HTTPException(
            status_code=400,
            detail="Provide extraction_json (body) or upload a PDF file.",
        )
    try:
        if extraction_json is not None:
            # direct path: sectionize given extraction JSON
            out = process_resume_payload(
                extraction_json=extraction_json, use_model_fallback=use_model_fallback
            )
            return JSONResponse(content=out)
        # else file upload
        if file:
            # verify file type heuristically
            filename = file.filename or "uploaded_file"
            content_type = file.content_type or ""
            if not (filename.lower().endswith(".pdf") or "pdf" in content_type.lower()):
                raise HTTPException(
                    status_code=400, detail="Only PDF resumes are supported."
                )
            content = await file.read()
            out = process_resume_payload(
                pdf_bytes=content,
                filename=filename,
                use_model_fallback=use_model_fallback,
            )
            return JSONResponse(content=out)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
