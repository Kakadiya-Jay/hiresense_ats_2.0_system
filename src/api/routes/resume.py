# src/api/routes/resume.py
"""
FastAPI route for resume processing.

Endpoint: POST /process_resume

Accepts either:
 - JSON body with "extraction_json": already-run Phase-1 extraction JSON, OR
 - multipart form upload with "file" (PDF) to run phase-1 extraction and then sectioning.

Optional query/body parameters:
 - use_model_fallback (bool): passed to sectioning
 - run_feature_extraction (bool): whether to run phase-3 feature extraction (default True)
 - jd_text (str): optional job description text forwarded to feature extraction for JD similarity

Returns:
 - JSON with 'sectioned' (phase-2 output) and 'features' (phase-3 output, if run)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Query
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
    use_model_fallback: bool = Query(True),
    run_feature_extraction: bool = Query(True),
    jd_text: Optional[str] = Body(None),
):
    """
    Route entrypoint for processing a resume. Either 'extraction_json' (phase-1)
    or 'file' (PDF) must be provided.
    """
    if extraction_json is None and file is None:
        raise HTTPException(
            status_code=400,
            detail="Provide extraction_json (body) or upload a PDF file.",
        )

    try:
        if extraction_json is not None:
            # direct path: sectionize given extraction JSON then optionally run feature extraction
            out = process_resume_payload(
                extraction_json=extraction_json,
                use_model_fallback=use_model_fallback,
                run_feature_extraction=run_feature_extraction,
                jd_text=jd_text,
            )
            return JSONResponse(content=out)

        # else file upload path
        if file:
            filename = file.filename or "uploaded_file"
            content_type = (file.content_type or "").lower()
            # accept only PDFs
            if not (filename.lower().endswith(".pdf") or "pdf" in content_type):
                raise HTTPException(
                    status_code=400, detail="Only PDF resumes are supported."
                )
            pdf_bytes = await file.read()
            out = process_resume_payload(
                pdf_bytes=pdf_bytes,
                filename=filename,
                use_model_fallback=use_model_fallback,
                run_feature_extraction=run_feature_extraction,
                jd_text=jd_text,
            )
            return JSONResponse(content=out)

        # should never reach here
        raise HTTPException(status_code=400, detail="No valid input provided.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
