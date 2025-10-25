# src/api/routes/resume.py
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
from typing import Optional, List, Tuple, Dict, Any
import traceback
import logging

from src.api.services.resume_service import (
    save_upload_bytes,
    process_pdf,
    get_stored_doc,
    match_job_description,
    score_resume,
    score_batch_by_doc_ids,
    process_and_score_files_batch,
    STORE,
)

from src.phases.scoring.helpers.scorer import score_candidate

router = APIRouter(prefix="/resume", tags=["resume"])
logger = logging.getLogger("hiresense.routes.resume")
logger.setLevel(logging.INFO)


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


from src.api.services.resume_service import _features_to_candidate_json


@router.post("/score_resume")
def score_resume_route(req: ScoreRequest):
    """
    Robust score_resume handler.

    Behavior (improved):
      - Fetch stored processed doc by doc_id (via get_stored_doc / STORE).
      - If stored candidate_json/features empty:
          1) If stored contains 'path' to the saved PDF -> re-run process_pdf(path, persist=True)
             (this re-run is the most robust way to re-generate features exactly how process_resume does)
          2) Else, try extract_features_from_sectioned on stored['sections'] (fallback)
      - After regeneration attempt, if candidate_json still empty -> return helpful HTTP 422 with debug
      - If candidate_json present -> call score_candidate(...) and return result (with doc_id + debug appended)
    """
    # Validate and fetch stored document
    if not req.doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required")

    # Try to fetch via get_stored_doc if available; else fallback to STORE dict
    try:
        stored = get_stored_doc(req.doc_id)
    except Exception:
        stored = STORE.get(req.doc_id) if isinstance(STORE, dict) else None

    if not stored:
        raise HTTPException(status_code=404, detail=f"doc_id {req.doc_id} not found")

    debug_msgs = []
    # Try candidate_json from stored doc (preferred)
    candidate_json = stored.get("candidate_json") or {}
    features_block = stored.get("features") or {}

    if candidate_json:
        debug_msgs.append("Found candidate_json in stored doc (using existing).")
    elif features_block:
        # convert features -> candidate_json using helper
        try:
            candidate_json = _features_to_candidate_json(features_block)
            if candidate_json:
                debug_msgs.append("Converted stored 'features' -> 'candidate_json'.")
                # persist back for next calls
                try:
                    stored["candidate_json"] = candidate_json
                    if isinstance(STORE, dict):
                        STORE[req.doc_id] = stored
                except Exception:
                    debug_msgs.append(
                        "Could not persist candidate_json back into STORE (non-fatal)."
                    )
        except Exception as e:
            debug_msgs.append(f"Conversion features->candidate_json failed: {e}")
            candidate_json = {}

    # If still empty, try reprocessing from saved file path (most reliable)
    if not candidate_json:
        saved_path = stored.get("path") or stored.get("file_path") or None
        if saved_path:
            try:
                debug_msgs.append(
                    "candidate_json missing: re-running process_pdf on saved file path to regenerate features."
                )
                # Re-run full processing. persist=True ensures STORE is updated with regenerated output
                reproc = process_pdf(saved_path, persist=True)
                # update stored reference to new result
                stored = reproc or stored
                candidate_json = stored.get("candidate_json") or {}
                if candidate_json:
                    debug_msgs.append(
                        "Reprocessing succeeded and candidate_json regenerated from PDF."
                    )
                else:
                    debug_msgs.append(
                        "Reprocessing completed but candidate_json still empty."
                    )
            except Exception as e:
                debug_msgs.append(f"Reprocessing via process_pdf failed: {e}")

    # If still empty, fall back to extracting from 'sections' using extractor (lightweight)
    if not candidate_json:
        try:
            # import here to avoid top-level cycles
            from src.phases.feature_extraction.main import (
                extract_features_from_sectioned,
            )

            sections = stored.get("sections") or {}
            if sections:
                debug_msgs.append(
                    "candidate_json missing: attempting extract_features_from_sectioned(sections) as fallback."
                )
                try:
                    features_extracted = extract_features_from_sectioned(
                        {"sections": sections}
                    )
                    candidate_json = (
                        _features_to_candidate_json(features_extracted) or {}
                    )
                    if candidate_json:
                        debug_msgs.append(
                            "Fallback extractor returned candidate_json via sections."
                        )
                        # persist back for future calls
                        stored["features"] = features_extracted
                        stored["candidate_json"] = candidate_json
                        if isinstance(STORE, dict):
                            STORE[req.doc_id] = stored
                    else:
                        debug_msgs.append(
                            "Fallback extractor returned no candidate_json (empty)."
                        )
                except Exception as e:
                    debug_msgs.append(f"extract_features_from_sectioned failed: {e}")
            else:
                debug_msgs.append(
                    "No 'sections' available in stored doc; cannot run fallback extractor."
                )
        except Exception as import_exc:
            debug_msgs.append(f"Could not import extractor fallback: {import_exc}")

    # After all attempts, if candidate_json still empty return informative response (422)
    if not candidate_json or (
        isinstance(candidate_json, dict)
        and all((not v) for v in candidate_json.values())
    ):
        # attach debug info for troubleshooting
        explanation = {
            "detail": "No candidate features found after regeneration attempts.",
            "debug": debug_msgs,
            "note": "Ensure process_resume persisted 'candidate_json' and 'features' to STORE[doc_id], or that the saved file still exists at stored['path'].",
        }
        # Instead of returning zero-scores silently, return 422 to force caller to inspect extraction step
        raise HTTPException(status_code=422, detail=explanation)

    # Ensure we have job description
    job_description = req.job_description if hasattr(req, "job_description") else None
    if not job_description:
        raise HTTPException(status_code=400, detail="job_description is required")

    # Call scorer
    try:
        from src.phases.scoring.helpers.scorer import score_candidate as scorer_fn

        score_result = scorer_fn(candidate_json, job_description)
        # include doc_id and debug info in the response for traceability
        score_result["doc_id"] = req.doc_id
        score_result.setdefault("debug", [])
        # append our handler debug messages (non-sensitive)
        score_result["debug"].extend(debug_msgs)
        # persist final scored snapshot back to STORE for audit if desired
        try:
            stored.setdefault("last_score", {})["snapshot"] = score_result
            if isinstance(STORE, dict):
                STORE[req.doc_id] = stored
        except Exception:
            # non-fatal
            pass
        return JSONResponse(content=score_result)
    except Exception as sc_exc:
        # return an explanatory error so you can see stacktrace during dev
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500, detail=f"Scoring failed: {str(sc_exc)}\n{tb}"
        )


# ---------------------------
# Batch scoring endpoints
# ---------------------------


class ScoreBatchRequest(BaseModel):
    doc_ids: List[str]
    job_description: str
    required_keywords: Optional[str] = None  # comma-separated
    top_k: Optional[int] = 5


def _validate_and_get_topk(value: Optional[int]) -> int:
    """Return a safe top_k integer (>=1)"""
    try:
        if value is None:
            return 1
        top_k = int(value)
        if top_k < 1:
            return 1
        return top_k
    except Exception:
        return 1


def _sort_and_select_topk(
    results: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """
    Sort results by (final_score, mean_top_scores, keyword_bonus) descending
    and return top_k slice (or fewer if not enough docs).
    """

    def sort_key(r: Dict[str, Any]):
        try:
            return (
                float(r.get("final_score", 0.0)),
                float(r.get("mean_top_scores", 0.0)),
                float(r.get("keyword_bonus", 0.0)),
            )
        except Exception:
            return (0.0, 0.0, 0.0)

    sorted_results = sorted(results, key=sort_key, reverse=True)
    return sorted_results[:top_k]


def _limit_matches_per_doc(results: List[Dict[str, Any]], max_matches: int = 3) -> None:
    """
    In-place: for each candidate result, keep only top 'max_matches' sorted by combined_score.
    """
    for r in results:
        matches = r.get("matches", [])
        if not isinstance(matches, list):
            r["matches"] = []
            continue
        try:
            matches_sorted = sorted(
                matches, key=lambda m: float(m.get("combined_score", 0.0)), reverse=True
            )
            r["matches"] = matches_sorted[:max_matches]
        except Exception:
            r["matches"] = matches[:max_matches]


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

    This endpoint:
      - scores all provided doc_ids (SEQUENTIALLY to avoid concurrent model init)
      - sorts by final_score desc (tie-breakers applied)
      - returns only top_k candidate documents
      - limits matches per doc to a small number for compact responses
    """

    # Basic validation
    if not req.doc_ids or len(req.doc_ids) == 0:
        raise HTTPException(
            status_code=400,
            detail="doc_ids is required and must contain at least one ID",
        )

    # prepare keywords list
    keywords = (
        [k.strip() for k in req.required_keywords.split(",")]
        if req.required_keywords
        else None
    )

    top_k = _validate_and_get_topk(req.top_k)
    max_matches = 3  # you can expose this as a parameter if you want

    errors: List[str] = []
    results: List[Dict[str, Any]] = []

    # -----------------------------
    # Hotfix: SEQUENTIAL processing
    # We do not use ThreadPoolExecutor here because the scoring pipeline
    # currently constructs a SentenceTransformer inside SemanticMatcher,
    # and parallel initialization caused a PyTorch "meta tensor" error.
    # Sequential scoring avoids concurrent model inits and solves the error.
    # -----------------------------
    for doc_id in req.doc_ids:
        try:
            scored = _score_single_doc_wrapper(doc_id, req.job_description, keywords)
            # Ensure minimal fields
            scored.setdefault("doc_id", doc_id)
            scored.setdefault("final_score", float(scored.get("final_score", 0.0)))
            scored.setdefault(
                "mean_top_scores", float(scored.get("mean_top_scores", 0.0))
            )
            scored.setdefault("keyword_bonus", float(scored.get("keyword_bonus", 0.0)))
            scored.setdefault("matches", scored.get("matches", []))
            results.append(scored)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("Error scoring doc_id=%s: %s\n%s", doc_id, exc, tb)
            errors.append(f"{doc_id}: {str(exc)}")

    # Sort & select top_k documents
    results_topk = _sort_and_select_topk(results, top_k=top_k)

    # Limit matches per document
    _limit_matches_per_doc(results_topk, max_matches=max_matches)

    response = {"results": results_topk, "errors": errors, "count": len(results_topk)}
    return JSONResponse(content=response)


def _score_single_doc_wrapper(
    doc_id: str, job_description: str, required_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Wrap call to your existing score_resume function in a safe manner.
    Adjust call if your score_resume signature differs.
    Expected score_resume signature:
       score_resume(doc_id: str, job_description: str, required_keywords: Optional[List[str]] = None, top_k: Optional[int] = None) -> dict
    """
    try:
        # Call your real scoring function (synchronous). If your function is async,
        # you should adapt this wrapper to run it in an event loop or refactor to async.
        scored = score_resume(
            doc_id, job_description, required_keywords=required_keywords
        )
        if isinstance(scored, dict):
            return scored
        else:
            # If the service returns a numeric score, wrap it into expected dict
            try:
                return {
                    "doc_id": doc_id,
                    "final_score": float(scored),
                    "mean_top_scores": 0.0,
                    "keyword_bonus": 0.0,
                    "matches": [],
                }
            except Exception:
                return {
                    "doc_id": doc_id,
                    "final_score": 0.0,
                    "mean_top_scores": 0.0,
                    "keyword_bonus": 0.0,
                    "matches": [],
                }
    except KeyError as ke:
        # propagate doc not found as a specific error entry
        logger.warning("Doc not found during scoring: %s", doc_id)
        return {
            "doc_id": doc_id,
            "final_score": 0.0,
            "mean_top_scores": 0.0,
            "keyword_bonus": 0.0,
            "matches": [],
        }
    except Exception as e:
        logger.exception("Exception in score_resume for doc_id=%s: %s", doc_id, str(e))
        return {
            "doc_id": doc_id,
            "final_score": 0.0,
            "mean_top_scores": 0.0,
            "keyword_bonus": 0.0,
            "matches": [],
        }


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
