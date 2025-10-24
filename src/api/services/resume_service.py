# src/api/services/resume_service.py
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
      - features (raw extractor output)  <-- NEW
      - candidate_json (simplified, scorer-friendly) <-- NEW
    """
    # Ingest raw text & pages
    ingested = extract_pdf_text(path)  # {"full_text", "pages", "page_blocks"}
    full_text = ingested.get("full_text", "")
    pages = ingested.get("pages", [])
    page_blocks = ingested.get("page_blocks", [])

    # Section heuristics
    sections = simple_section_split(full_text)

    # Sentence segmentation and BM25 tokenization
    sentences = sentences_from_text(full_text)
    tokenized = preprocess_for_bm25(full_text)

    # Prepare placeholders for features/candidate_json
    features_extracted = {}
    candidate_json = {}

    # ---------------------------
    # NEW: run feature extraction and produce simplified candidate_json
    # ---------------------------
    try:
        # import here to avoid import cycles at module import time
        from src.phases.feature_extraction.main import extract_features_from_sectioned
        # Build input for extractor — include sections/sentences/full_text
        sectioned_input = {
            "sections": sections,
            "sentences": sentences,
            "full_text": full_text,
            "source": "resume_pdf"
        }
        # Run feature extractor (it typically returns a dict with 'features' key)
        features_extracted = extract_features_from_sectioned(sectioned_input) or {}
        # Map to simplified candidate_json that scorer expects
        # _features_to_candidate_json should be defined in this module (or import from helper)
        candidate_json = _features_to_candidate_json(features_extracted)
    except Exception as _ex:
        # If extraction fails, keep candidate_json empty but don't crash the pipeline
        features_extracted = features_extracted or {}
        candidate_json = candidate_json or {}
        # Try to log the exception if a logger is available
        try:
            _logger = globals().get("logger")
            if _logger:
                _logger.warning("Feature extraction failed for %s: %s", path, str(_ex))
        except Exception:
            pass
    # ---------------------------

    # Create new doc_id and result
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
        # persisted extractor outputs for later scoring
        "features": features_extracted,
        "candidate_json": candidate_json,
    }

    if persist:
        try:
            STORE[doc_id] = result
        except Exception:
            # fallback if STORE is a custom interface (not a plain dict) — try to call set/save method
            try:
                save_fn = globals().get("save_doc") or globals().get("store_doc")
                if callable(save_fn):
                    save_fn(doc_id, result)
                else:
                    # last-resort: ignore persistence failure but inform via logger if present
                    _logger = globals().get("logger")
                    if _logger:
                        _logger.warning("Failed to persist processed doc to STORE for %s", doc_id)
            except Exception:
                pass

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


def _features_to_candidate_json(feature_extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map the feature extractor output (feature_extracted) to the simplified candidate_json
    expected by the scorer. Defensive: works with several extractor output shapes.

    Returns a dict with keys:
      - skills: List[str]
      - projects: List[{"title": str, "summary": str}]
      - experience: List[str]
      - education: List[str]
      - certifications: List[str]
      - open_source: List[str]
    """
    cand: Dict[str, Any] = {}
    feat = (
        feature_extracted.get("features", {})
        if isinstance(feature_extracted, dict)
        else {}
    )

    # SKILLS
    skills: List[str] = []
    skills_dict = feat.get("skills", {}) or {}
    # skills_dict might be {"technical": [...], "soft": [...]} or a flat list
    if isinstance(skills_dict, dict):
        for subcat, items in skills_dict.items():
            if not items:
                continue
            for it in items:
                if isinstance(it, dict):
                    name = it.get("name") or it.get("skill") or it.get("title") or ""
                else:
                    name = str(it)
                if name:
                    skills.append(name)
    else:
        # if skills_dict is a list
        for it in skills_dict or []:
            skills.append(it.get("name") if isinstance(it, dict) else str(it))

    cand["skills"] = skills

    # PROJECTS
    projects_list: List[Dict[str, str]] = []
    projects_block = feat.get("projects", {}) or {}
    possible_projects = (
        projects_block.get("projects") or projects_block.get("project_list") or []
    )
    # If extractor places projects at top level
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
    # fallback if extractor stores a list in feat['work_experience'] or similar
    if not possible_exps:
        possible_exps = feat.get("work_experience") or feat.get("roles") or []

    for e in possible_exps or []:
        if isinstance(e, dict):
            # try to produce a readable summary
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

    return cand


# ---------------------------
# End helper
# ---------------------------
