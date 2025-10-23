#src/api/services/resume_service.py
"""
Resume service - orchestration layer.
Added batch scoring helpers: score_batch_by_doc_ids and process_and_score_files_batch.
"""

import os
import uuid
from typing import Dict, Any, List, Optional, Tuple

# Import pipeline pieces (adjust import paths if you place helpers elsewhere)
from src.phases.text_extraction.helpers.pdf_ingest import extract_pdf_text
from src.phases.text_extraction.helpers.preprocess import (
    simple_section_split,
    sentences_from_text,
    preprocess_for_bm25,
    tokens_for_sentence,
)
from src.phases.embeddings_matching.helpers.bm25_matcher import BM25Matcher
from src.phases.embeddings_matching.helpers.semantic_matcher import SemanticMatcher
from src.phases.scoring.helpers.scoring import (
    combine_scores,
)

# For a simple in-memory store (demo). Replace with DB/FS for production.
STORE: Dict[str, Dict[str, Any]] = {}
UPLOAD_DIR = os.environ.get("HIRE_SENSE_UPLOAD_DIR", "/tmp/hiresense_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload_bytes(file_bytes: bytes, filename_hint: Optional[str] = None) -> str:
    """Save uploaded bytes to a local file and return path."""
    file_id = str(uuid.uuid4())
    # keep .pdf extension
    safe_hint = filename_hint.replace(" ", "_") if filename_hint else "upload.pdf"
    name = f"{file_id}_{safe_hint}"
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def process_pdf(path: str, persist: bool = True) -> Dict[str, Any]:
    """
    Run ingestion + preprocessing on given PDF file path.
    Returns a dictionary with:
      - doc_id
      - full_text
      - pages
      - page_blocks
      - sections (heuristic)
      - sentences (list)
      - tokenized (for bm25: list[list[str]])
    """
    ingested = extract_pdf_text(path)  # {"full_text", "pages", "page_blocks"}
    full_text = ingested.get("full_text", "")
    pages = ingested.get("pages", [])
    page_blocks = ingested.get("page_blocks", [])

    # Section heuristics
    sections = simple_section_split(full_text)

    # Sentence segmentation and BM25 tokenization
    sentences = sentences_from_text(full_text)
    tokenized = preprocess_for_bm25(full_text)

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
    }

    if persist:
        STORE[doc_id] = result
    return result


def get_stored_doc(doc_id: str) -> Optional[Dict[str, Any]]:
    return STORE.get(doc_id)


def match_job_description(doc_id: str, job_description: str, top_k: int = 5) -> Dict:
    """
    Perform BM25 + semantic matching for stored doc doc_id against job_description,
    combine scores and return result list.

    This implementation is robust: it accepts scorer outputs that use either
    'combined_score' or 'score' as the combined-score field and normalizes
    results to always include both keys.
    """
    doc = get_stored_doc(doc_id)
    if not doc:
        raise KeyError(f"doc_id {doc_id} not found")

    # BM25 on tokenized sentences
    bm25 = BM25Matcher(doc["tokenized"])
    q_tokens = tokens_for_sentence(job_description)
    bm25_res = bm25.query(q_tokens, top_k=top_k)

    # Semantic matcher on sentence texts
    sem = SemanticMatcher(doc["sentences"])
    sem_res = sem.query(job_description, top_k=top_k)

    # combine_bm25_semantic returns dicts that may use key names like 'combined_score'
    combined = combine_scores(bm25_res, sem_res, w_bm25=0.5, w_sem=0.5, top_k=top_k)

    # Map combined to sentences + explanatory fields.
    # Be tolerant: some implementations return 'combined_score', others return 'score'
    matches = []
    for item in combined:
        idx = item.get("idx")
        # read value from whichever key exists
        combined_score = item.get(
            "combined_score", item.get("score", item.get("combined", 0.0))
        )
        # also expose consistent "score" field for older callers
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
                # expose both keys (backwards & forwards compat)
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
    """
    Aggregate top-k match scores into a final score and include small heuristics (keyword coverage bonus).

    This function is tolerant to either 'combined_score' or 'score' fields in match items.
    """
    match_res = match_job_description(doc_id, job_description, top_k=top_k)
    # use whichever key exists, but prefer 'score'
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


# ----------------------------
# New: Batch helpers
# ----------------------------


def score_batch_by_doc_ids(
    doc_ids: List[str],
    job_description: str,
    required_keywords: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Score multiple existing stored docs by doc_id.
    Returns dict with 'results' (list of per-doc results) and 'errors' (list of error entries).
    """
    results = []
    errors = []
    for did in doc_ids:
        try:
            if not get_stored_doc(did):
                raise KeyError(f"doc_id {did} not found")
            scored = score_resume(
                did, job_description, required_keywords=required_keywords, top_k=top_k
            )
            results.append(scored)
        except Exception as e:
            errors.append({"doc_id": did, "error": str(e)})
    return {"results": results, "errors": errors, "count": len(results)}


def process_and_score_files_batch(
    uploaded_files: List[Tuple[bytes, str]],
    job_description: str,
    required_keywords: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Accepts a list of tuples (file_bytes, filename_hint).
    For each file:
      - save bytes
      - process PDF (process_pdf)
      - score using score_resume
    Returns results + errors.
    """
    results = []
    errors = []
    for file_bytes, filename in uploaded_files:
        try:
            saved_path = save_upload_bytes(file_bytes, filename_hint=filename)
            processed = process_pdf(saved_path, persist=True)
            doc_id = processed["doc_id"]
            scored = score_resume(
                doc_id,
                job_description,
                required_keywords=required_keywords,
                top_k=top_k,
            )
            results.append(scored)
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})
    return {"results": results, "errors": errors, "count": len(results)}
