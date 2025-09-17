# src/api/routes/resume.py
from fastapi import APIRouter, UploadFile, File
import tempfile, os
from src.pipeline.pipeline import run_pipeline_for_pdf

router = APIRouter()


@router.post("/process_resume")
async def process_resume(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()
        result = run_pipeline_for_pdf(tmp.name)
        candidate_meta = result.get("candidate", {})
        # return only Phase1 outputs required for demo
        return {
            "cleaned_text_updated": candidate_meta.get("cleaned_text_updated"),
            "link_metadata": candidate_meta.get("link_metadata"),
            "emails": candidate_meta.get("emails", []),
            "phones": candidate_meta.get("phones", []),
        }
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
