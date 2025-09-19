# src/api/routes/resume.py
"""
Resume processing route (robust).

Supports:
  - multipart/form-data with UploadFile 'file' (preferred).
  - JSON body with 'raw_sections' OR 'cleaned_text_updated' (backward-compatible).
  - JSON body with 'file_bytes_b64' (base64-encoded PDF bytes) — supported but not recommended.

Behavior:
  - Normalizes simple form fields (JSON strings inside form fields).
  - Calls process_resume_payload(payload, file_bytes=...) and returns {"candidate": ...}
  - Returns 400 on bad client input (ValueError from service) and 500 on unexpected server errors.
"""

import json
import base64
from typing import Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.logger import logger

from ..services.resume_service import process_resume_payload

router = APIRouter()


def parse_json_field(value: Optional[str]):
    """Parse a form field which may contain JSON string. Return native type or original string."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def _decode_b64_string_to_bytes(b64: str) -> Optional[bytes]:
    if not b64:
        return None
    try:
        if b64.startswith("data:"):
            # strip prefix like "data:application/pdf;base64,<data>"
            parts = b64.split(",", 1)
            if len(parts) == 2:
                b64 = parts[1]
        return base64.b64decode(b64)
    except Exception as ex:
        logger.warning("Failed to decode base64 file bytes: %s", ex)
        return None


@router.post("/process_resume")
async def process_resume(
    request: Request,
    file: Optional[UploadFile] = File(None),
    # allow optional form fields when using multipart/form-data
    raw_sections: Optional[str] = Form(None),
    section_confidence: Optional[str] = Form(None),
    link_metadata: Optional[str] = Form(None),
    emails: Optional[str] = Form(None),
    phones: Optional[str] = Form(None),
):
    """
    Main route entry for resume processing.
    """
    try:
        # Preferred flow: multipart upload with file
        if file is not None:
            file_bytes = await file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            # parse optional form fields (they may be JSON-encoded strings)
            payload: Dict[str, Any] = {
                "raw_sections": parse_json_field(raw_sections),
                "section_confidence": parse_json_field(section_confidence) or {},
                "link_metadata": parse_json_field(link_metadata) or [],
                "emails": parse_json_field(emails) or [],
                "phones": parse_json_field(phones) or [],
            }

            # Call service with file_bytes
            candidate = process_resume_payload(payload, file_bytes=file_bytes)
            return {"candidate": candidate}

        # No multipart file: try JSON body (backwards compatible or b64-encoded bytes)
        content_type = request.headers.get("content-type", "")
        # Attempt to read JSON body safely
        try:
            body = await request.json()
        except Exception:
            # Could be empty body or invalid json
            raise HTTPException(
                status_code=400,
                detail="Invalid or missing JSON body. If uploading a file, use multipart/form-data with 'file' field.",
            )

        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400, detail="Request JSON body must be an object/dict."
            )

        # If the client PUT base64 bytes inside JSON, decode them
        file_bytes = None
        if "file_bytes_b64" in body and body.get("file_bytes_b64"):
            file_bytes = _decode_b64_string_to_bytes(body.get("file_bytes_b64"))
            if file_bytes is None:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid base64 content in 'file_bytes_b64'.",
                )

        # If file_bytes provided via base64: use that preferred flow
        if file_bytes:
            candidate = process_resume_payload(body, file_bytes=file_bytes)
            return {"candidate": candidate}

        # else call service with provided payload (expects raw_sections or cleaned_text_updated)
        try:
            candidate = process_resume_payload(body)
            return {"candidate": candidate}
        except ValueError as ve:
            # This is a client error: missing required fields or malformed payload
            raise HTTPException(status_code=400, detail=str(ve))

    except HTTPException:
        # Re-raise HTTP errors to be handled by FastAPI
        raise
    except ValueError as ve:
        # Service raised a ValueError we didn't anticipate above: treat as 400
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as ex:
        # Unexpected server error
        logger.exception("Unhandled error in /process_resume: %s", ex)
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {ex}")
