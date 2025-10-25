# src/api/services/resume_service.py
"""
Resume service - orchestration layer.
Extended with batch persistence + background processing helpers.

Key additions:
 - STORE_BATCHES: in-memory mapping batch_id -> metadata/status (swap to DB in production)
 - process_and_score_files_batch_background: background worker used by FastAPI BackgroundTasks
 - score_batch_by_doc_ids: optionally concurrent scoring (env toggle)
"""

import os
import uuid
import re
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json

logger = logging.getLogger("hiresense.resume_service")
logger.setLevel(logging.INFO)

# existing imports (keep as in your project)
from src.phases.text_extraction.helpers.pdf_ingest import extract_pdf_text
from src.phases.text_extraction.helpers.preprocess import (
    simple_section_split,
    sentences_from_text,
    preprocess_for_bm25,
    tokens_for_sentence,
)
from src.phases.embeddings_matching.helpers.bm25_matcher import BM25Matcher
from src.phases.embeddings_matching.helpers.semantic_matcher import SemanticMatcher
from src.phases.scoring.helpers.scoring import combine_scores

# scorer used by score_resume (keep existing scorer import in runtime path)
try:
    from src.phases.scoring.helpers.scorer import score_candidate
except Exception:
    # Fallback stub for dev-time to avoid import errors; replace with real scorer
    def score_candidate(candidate_json, jd_text, *args, **kwargs):
        return {
            "final_score": 0.0,
            "raw_score": 0.0,
            "denom": 1.0,
            "per_feature": {},
            "evidence": {},
        }


# In-memory stores (replace with DB/Redis for production)
STORE: Dict[str, Dict[str, Any]] = {}
STORE_BATCHES: Dict[str, Dict[str, Any]] = {}

UPLOAD_DIR = os.environ.get("HIRE_SENSE_UPLOAD_DIR", "/tmp/hiresense_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# env toggle: allow concurrent scoring (be careful: only enable after verifying model thread-safety)
ALLOW_CONCURRENT_SCORING = bool(
    os.environ.get("HIRE_SENSE_ALLOW_CONCURRENT_SCORING", "").strip()
)


# ----------------------
# Utilities: persistence
# ----------------------
def persist_doc(doc: Dict[str, Any]) -> None:
    """Persist processed doc into in-memory STORE (or DB in prod)."""
    if not isinstance(doc, dict) or "doc_id" not in doc:
        return
    STORE[doc["doc_id"]] = doc


def get_stored_doc(doc_id: str) -> Optional[Dict[str, Any]]:
    return STORE.get(doc_id)


# ----------------------
# Batch helpers
# ----------------------
def persist_batch(batch_id: str, meta: Dict[str, Any]) -> None:
    STORE_BATCHES[batch_id] = meta


def get_batch(batch_id: str) -> Optional[Dict[str, Any]]:
    return STORE_BATCHES.get(batch_id)


def update_batch_status(batch_id: str, status: Optional[str]) -> None:
    if not batch_id:
        return
    b = STORE_BATCHES.get(batch_id, {})
    b["status"] = status
    b["updated_at"] = time.time()
    STORE_BATCHES[batch_id] = b


def append_doc_to_batch(batch_id: str, doc_id: str) -> None:
    if not batch_id:
        return
    b = STORE_BATCHES.get(
        batch_id, {"doc_ids": [], "status": "processing", "created_at": time.time()}
    )
    docs = b.get("doc_ids", [])
    if doc_id not in docs:
        docs.append(doc_id)
    b["doc_ids"] = docs
    b["updated_at"] = time.time()
    STORE_BATCHES[batch_id] = b


def persist_batch_result(batch_id: str, result: Dict[str, Any]) -> None:
    if not batch_id:
        return
    b = STORE_BATCHES.get(batch_id, {})
    b["last_result"] = result
    b["updated_at"] = time.time()
    STORE_BATCHES[batch_id] = b


# -------------------------
# File utilities
# -------------------------
def save_upload_bytes(file_bytes: bytes, filename_hint: Optional[str] = None) -> str:
    """Save uploaded bytes to a local file and return path."""
    file_id = str(uuid.uuid4())
    safe_hint = filename_hint.replace(" ", "_") if filename_hint else "upload.pdf"
    name = f"{file_id}_{safe_hint}"
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


# -------------------------
# Existing single-file pipeline
# -------------------------
def process_pdf(path: str, persist: bool = True) -> Dict[str, Any]:
    """
    Run ingestion + preprocessing on given PDF file path.
    Returns a dictionary with:
      - doc_id, path, full_text, pages, page_blocks, sections, sentences, tokenized
      - features (raw extractor output)
      - candidate_json (simplified scorer-friendly)
    """
    # Ingest raw text & pages
    ingested = extract_pdf_text(path)  # {"full_text", "pages", "page_blocks"}
    full_text = ingested.get("full_text", "") or ""
    pages = ingested.get("pages", []) or []
    page_blocks = ingested.get("page_blocks", []) or []

    # Section heuristics
    sections = simple_section_split(full_text) or {}

    # Sentence segmentation and tokenization
    sentences = sentences_from_text(full_text) or []
    tokenized = preprocess_for_bm25(full_text) or []

    # Prepare placeholders
    features_extracted = {}
    candidate_json = {}

    # Feature extraction (try/catch to avoid crashing)
    try:
        from src.phases.feature_extraction.main import extract_features_from_sectioned

        sectioned_input = {
            "sections": sections,
            "sentences": sentences,
            "full_text": full_text,
            "source": "resume_pdf",
        }
        features_extracted = extract_features_from_sectioned(sectioned_input) or {}
        candidate_json = _features_to_candidate_json(features_extracted)
    except Exception as _ex:
        # keep features_extracted/candidate_json as empty dicts
        logger.warning("Feature extraction failed for %s: %s", path, str(_ex))

    doc_id = str(uuid.uuid4())
    result = {
        "doc_id": doc_id,
        "path": path,
        "full_text": full_text,
        "pages": pages,
        "page_blocks": page_blocks,
        "sections": sections,
        "sentences": sentences,
        "tokenized": tokenized,
        "features": features_extracted,
        "candidate_json": candidate_json,
    }

    if persist:
        persist_doc(result)

    return result


# -------------------------
# Matching & scoring (existing)
# -------------------------
def match_job_description(doc_id: str, job_description: str, top_k: int = 5) -> Dict:
    doc = get_stored_doc(doc_id)
    if not doc:
        raise KeyError(f"doc_id {doc_id} not found")

    bm25 = BM25Matcher(doc["tokenized"])
    q_tokens = tokens_for_sentence(job_description)
    bm25_res = bm25.query(q_tokens, top_k=top_k)

    sem = SemanticMatcher(doc["sentences"])
    sem_res = sem.query(job_description, top_k=top_k)

    combined = combine_scores(bm25_res, sem_res, w_bm25=0.5, w_sem=0.5, top_k=top_k)

    matches = []
    for item in combined:
        idx = item.get("idx")
        combined_score = item.get(
            "combined_score", item.get("score", item.get("combined", 0.0))
        )
        score_val = float(combined_score)
        matches.append(
            {
                "sentence_idx": idx,
                "sentence": (
                    doc["sentences"][idx]
                    if idx is not None and idx < len(doc["sentences"])
                    else ""
                ),
                "bm25_raw": item.get("bm25_raw", 0.0),
                "sem_raw": item.get("sem_raw", 0.0),
                "bm25_norm": item.get("bm25_norm", 0.0),
                "sem_norm": item.get("sem_norm", 0.0),
                "combined_score": score_val,
                "score": score_val,
            }
        )
    return {"doc_id": doc_id, "matches": matches, "top_k": top_k}


def score_resume(
    doc_id: str,
    job_description: str,
    required_keywords: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict:
    match_res = match_job_description(doc_id, job_description, top_k=top_k)
    top_scores = []
    for m in match_res["matches"]:
        if "score" in m:
            top_scores.append(float(m["score"]))
        elif "combined_score" in m:
            top_scores.append(float(m["combined_score"]))
        else:
            top_scores.append(0.0)
    mean_top = float(sum(top_scores) / max(1, len(top_scores)))

    keyword_bonus = 0.0
    if required_keywords:
        full_text_low = get_stored_doc(doc_id)["full_text"].lower()
        found = sum(1 for k in required_keywords if k.strip().lower() in full_text_low)
        keyword_bonus = (found / max(1, len(required_keywords))) * 0.1

    final_score = min(1.0, mean_top + keyword_bonus)
    return {
        "doc_id": doc_id,
        "final_score": final_score,
        "mean_top_scores": mean_top,
        "keyword_bonus": keyword_bonus,
        "matches": match_res["matches"],
    }


# -------------------------
# Batch: processing in background & scoring helpers
# -------------------------
# def process_and_score_files_batch_background(


def process_and_score_files_batch(
    uploaded_files: List[Tuple[bytes, str]],
    batch_id_or_job_description=None,
    job_description: Optional[str] = None,
    required_keywords: Optional[List[str]] = None,
    top_k: int = 5,
    pre_score_each: bool = False,
):
    """
    Backwards-compatible wrapper for batch processing.

    Accepts both calling styles:
      - process_and_score_files_batch(uploaded_files, job_description, required_keywords=..., top_k=...)
      - process_and_score_files_batch(uploaded_files, batch_id, job_description, required_keywords=..., top_k=...)

    If second param looks like a uuid-like string and job_description is provided,
    it is treated as batch_id. Otherwise, it is treated as job_description and a new
    batch_id will be created.

    Returns summary: {
      "batch_id": str,
      "doc_ids": [...],
      "results": [per-file entries],
      "errors": [...],
      "count": int
    }
    """
    # Normalize args to (batch_id, job_description)
    batch_id = None
    # If caller provided job_description via second positional arg (old call)
    if job_description is None and isinstance(batch_id_or_job_description, str):
        # scenario: called as process_and_score_files_batch(uploaded_files, job_description_string, ...)
        batch_id = str(uuid.uuid4())
        job_description = batch_id_or_job_description
    elif job_description is not None and isinstance(batch_id_or_job_description, str):
        # scenario: called as process_and_score_files_batch(uploaded_files, batch_id, job_description, ...)
        batch_id = batch_id_or_job_description
    else:
        # if nothing sensible, generate batch_id
        batch_id = str(uuid.uuid4())

    # Validate job_description
    if (
        not job_description
        or not isinstance(job_description, str)
        or not job_description.strip()
    ):
        raise ValueError("job_description is required and must be a non-empty string")

    # Initialize batch meta
    start = time.perf_counter()
    persist_batch(
        batch_id,
        {
            "status": "processing",
            "created_at": time.time(),
            "job_description": job_description,
            "required_keywords": required_keywords,
            "top_k": top_k,
            "doc_ids": [],
            "files": [],
        },
    )

    files_summary = []
    errors = []
    processed_count = 0

    for file_bytes, filename in uploaded_files:
        t0 = time.perf_counter()
        entry = {
            "filename": filename,
            "status": "pending",
            "doc_id": None,
            "duration_s": None,
        }
        try:
            saved_path = save_upload_bytes(file_bytes, filename_hint=filename)
            processed = process_pdf(saved_path, persist=True)
            doc_id = processed.get("doc_id")
            if not doc_id:
                # defensive: ensure doc_id exists
                doc_id = str(uuid.uuid4())
                processed["doc_id"] = doc_id
                persist_doc(processed)
            entry["doc_id"] = doc_id
            entry["status"] = "processed"
            entry["duration_s"] = time.perf_counter() - t0
            append_doc_to_batch(batch_id, doc_id)
            persist_doc(processed)
            processed_count += 1

            if pre_score_each:
                try:
                    score_res = score_resume(
                        doc_id,
                        job_description,
                        required_keywords=required_keywords,
                        top_k=top_k,
                    )
                    entry["pre_score"] = {
                        "final_score": score_res.get("final_score"),
                        "mean_top_scores": score_res.get("mean_top_scores"),
                    }
                except Exception as se:
                    entry["pre_score_error"] = str(se)

        except Exception as e:
            tb = traceback.format_exc()
            logger.exception(
                "Error while processing uploaded file %s: %s", filename, tb
            )
            entry["status"] = "failed"
            entry["error"] = str(e)
            entry["trace"] = tb
            errors.append({"filename": filename, "error": str(e), "trace": tb})

        files_summary.append(entry)
        # update batch partial progress
        persist_batch(
            batch_id,
            {
                "status": "processing",
                "files": files_summary,
                "processed_count": processed_count,
                "errors": errors,
                "updated_at": time.time(),
            },
        )

    duration = time.perf_counter() - start
    final_summary = {
        "batch_id": batch_id,
        "files": files_summary,
        "errors": errors,
        "count": processed_count,
        "duration_s": duration,
        "created_at": get_batch(batch_id).get("created_at", time.time()),
    }
    persist_batch(
        batch_id,
        {"status": "ready", "summary": final_summary, "updated_at": time.time()},
    )
    # return in the shape caller expects
    return {
        "batch_id": batch_id,
        "doc_ids": [f.get("doc_id") for f in files_summary if f.get("doc_id")],
        "results": files_summary,
        "errors": errors,
        "count": processed_count,
    }


def _score_single_doc_worker(
    doc_id: str,
    job_description: str,
    required_keywords: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Concurrency-safe wrapper around score_resume. Returns standardized dict including score_result.
    """
    # guard: ensure stored doc exists
    stored = get_stored_doc(doc_id)
    if not stored:
        raise KeyError(f"doc_id {doc_id} not found")
    # call existing score_resume which returns a dict with final_score, mean_top_scores, keyword_bonus, matches
    scored = score_resume(
        doc_id, job_description, required_keywords=required_keywords, top_k=top_k
    )
    # optionally attach full score_result (detailed) by calling scorer if desired
    # If you want the internal per_feature breakdown from your scorer (score_candidate), you can call that here:
    try:
        candidate_json = stored.get("candidate_json") or {}
        # scorer may accept candidate_json OR the stored doc; try candidate_json first
        try:
            score_result_full = score_candidate(candidate_json, job_description)
        except Exception:
            # fallback to calling scorer on stored doc
            score_result_full = score_candidate(stored, job_description)
    except Exception:
        score_result_full = None

    return {
        "doc_id": doc_id,
        "final_score": float(scored.get("final_score", 0.0)),
        "mean_top_scores": float(scored.get("mean_top_scores", 0.0)),
        "keyword_bonus": float(scored.get("keyword_bonus", 0.0)),
        "matches": scored.get("matches", []),
        "score_result": score_result_full,
    }


def score_batch_by_doc_ids(
    doc_ids: List[str],
    job_description: str,
    required_keywords: Optional[List[str]] = None,
    top_k: int = 5,
    max_workers: int = 4,
    timeout_per_doc: int = 60,
) -> Dict[str, Any]:
    """
    Score a list of stored docs and return sorted results, optionally using concurrency.

    Default behaviour: sequential scoring (safe for models that don't like parallel init).
    If environment variable HIRE_SENSE_ALLOW_CONCURRENT_SCORING=1 is set, this will use ThreadPoolExecutor.
    """
    results = []
    errors = []
    start = time.perf_counter()

    # prefer sequential unless configured otherwise
    if ALLOW_CONCURRENT_SCORING and len(doc_ids) > 1:
        # Concurrent scoring path
        max_workers = min(max_workers or 4, max(1, len(doc_ids)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(
                    _score_single_doc_worker,
                    did,
                    job_description,
                    required_keywords,
                    top_k,
                ): did
                for did in doc_ids
            }
            for f in as_completed(future_map):
                did = future_map[f]
                try:
                    res = f.result(timeout=timeout_per_doc)
                    results.append(res)
                except Exception as e:
                    errors.append(
                        {
                            "doc_id": did,
                            "error": str(e),
                            "trace": traceback.format_exc(),
                        }
                    )
    else:
        # Sequential scoring (safe default)
        for did in doc_ids:
            try:
                res = _score_single_doc_worker(
                    did, job_description, required_keywords, top_k
                )
                results.append(res)
            except Exception as e:
                errors.append(
                    {"doc_id": did, "error": str(e), "trace": traceback.format_exc()}
                )

    # sort results descending by final_score
    results_sorted = sorted(
        results, key=lambda r: float(r.get("final_score", 0.0)), reverse=True
    )

    response = {
        "results": results_sorted,
        "errors": errors,
        "count": len(results_sorted),
        "duration_s": time.perf_counter() - start,
    }
    return response


# -------------------------
# Fallback _features_to_candidate_json (retain your existing mapping)
# -------------------------
def _features_to_candidate_json(feature_extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map the extractor output into simplified candidate_json expected by scorer.
    Works with both shapes: {'features': {...}} or features dict directly.
    """
    cand: Dict[str, Any] = {}
    feat = {}
    if isinstance(feature_extracted, dict):
        if "features" in feature_extracted and isinstance(
            feature_extracted["features"], dict
        ):
            feat = feature_extracted["features"]
        else:
            feat = feature_extracted
    else:
        feat = {}

    # SKILLS
    skills: List[str] = []
    skills_dict = feat.get("skills", {}) or {}
    if isinstance(skills_dict, dict):
        for subcat, items in skills_dict.items():
            if not items:
                continue
            for it in items or []:
                if isinstance(it, dict):
                    name = it.get("name") or it.get("skill") or it.get("title") or ""
                else:
                    name = str(it)
                if name:
                    skills.append(name)
    else:
        for it in skills_dict or []:
            skills.append(it.get("name") if isinstance(it, dict) else str(it))
    cand["skills"] = skills

    # PROJECTS
    projects_list: List[Dict[str, str]] = []
    projects_block = feat.get("projects", {}) or {}
    possible_projects = (
        projects_block.get("projects") or projects_block.get("project_list") or []
    )
    if not possible_projects and isinstance(feat.get("projects"), list):
        possible_projects = feat.get("projects")
    for p in possible_projects or []:
        if isinstance(p, dict):
            title = (
                p.get("title")
                or p.get("name")
                or (p.get("summary")[:60] if p.get("summary") else "")
            )
            summary = (
                p.get("summary") or p.get("description") or p.get("evidence") or ""
            )
            projects_list.append({"title": title, "summary": summary})
        else:
            projects_list.append({"title": str(p)[:80], "summary": ""})
    cand["projects"] = projects_list

    # EXPERIENCE
    exp_list: List[str] = []
    exp_block = feat.get("experience", {}) or {}
    possible_exps = exp_block.get("experience") or exp_block.get("positions") or []
    if not possible_exps:
        possible_exps = feat.get("work_experience") or feat.get("roles") or []
    for e in possible_exps or []:
        if isinstance(e, dict):
            summary = (
                e.get("summary")
                or e.get("description")
                or e.get("role")
                or e.get("company")
                or str(e)
            )
            exp_list.append(summary)
        else:
            exp_list.append(str(e))
    cand["experience"] = exp_list

    # EDUCATION
    edu_list: List[str] = []
    edu_block = feat.get("education", {}) or {}
    possible_edu = edu_block.get("education") or feat.get("education") or []
    for e in possible_edu or []:
        if isinstance(e, dict):
            degree = e.get("degree") or e.get("degree_name") or e.get("name") or str(e)
            edu_list.append(degree)
        else:
            edu_list.append(str(e))
    cand["education"] = edu_list

    # CERTIFICATIONS
    certs: List[str] = []
    cert_block = feat.get("certifications", {}) or {}
    cert_items = cert_block.get("certifications") or feat.get("certifications") or []
    for c in cert_items or []:
        if isinstance(c, dict):
            certs.append(
                c.get("certificate_title") or c.get("title") or c.get("name") or str(c)
            )
        else:
            certs.append(str(c))
    cand["certifications"] = certs

    # OPEN SOURCE / RESEARCH
    os_list: List[str] = []
    os_block = feat.get("open_source", {}) or {}
    os_items = (
        os_block.get("open_source")
        or feat.get("open_source")
        or feat.get("projects", [])
    )
    for it in os_items or []:
        if isinstance(it, dict):
            os_list.append(
                it.get("summary") or it.get("title") or it.get("name") or str(it)
            )
        else:
            os_list.append(str(it))
    cand["open_source"] = os_list

    research_items = feat.get("research_papers") or feat.get("research") or []
    research_list = []
    for r in research_items or []:
        if isinstance(r, dict):
            research_list.append(r.get("title") or r.get("summary") or str(r))
        else:
            research_list.append(str(r))
    cand["research"] = research_list

    return cand
