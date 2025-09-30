import streamlit as st
import requests
import json
import io
import zipfile
import time
from pathlib import Path
import pandas as pd
import os

# If you have a settings module, keep the import; otherwise this is harmless.
try:
    from src.core.config import settings
except Exception:
    settings = None


# ---- CONFIG: separate base URLs for Auth and Resume services ----
# --- safe config loading: prefer env vars, then st.secrets (if exists), then defaults ---
def _get_secret(key, default):
    # 1) environment variable (easy for CI / local override)
    val = os.getenv(key.upper())
    if val:
        return val
    # 2) streamlit secrets (only if present and contains key)
    try:
        # st.secrets behaves like a dict when secrets.toml is present
        s = st.secrets.get(key)
        if s:
            return s
    except Exception:
        # no secrets.toml present, ignore
        pass
    # 3) fallback default
    return default


AUTH_BASE = _get_secret("auth_api_base", "http://127.0.0.1:8000")
RESUME_BASE = _get_secret("resume_api_base", "http://127.0.0.1:8001")
TIMEOUT = int(_get_secret("request_timeout", 30))


# ---- Auth / helper utilities ----
def get_auth_headers():
    token = st.session_state.get("access_token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def handle_api_error(resp):
    """Return JSON on success; raise informative Exception on error"""
    if resp is None:
        raise Exception("No response object")
    if resp.status_code == 401:
        st.warning("Session expired or unauthorized. Please login again.")
        st.session_state.pop("access_token", None)
        raise Exception("401 Unauthorized")
    try:
        return resp.json()
    except ValueError:
        # non-json reply
        resp.raise_for_status()


# ---- API wrapper functions (resume endpoints) ----
def api_process_resume(file_bytes, filename, metadata=None):
    url = f"{RESUME_BASE}/resume/process_resume"
    files = {"file": (filename, file_bytes, "application/pdf")}
    data = metadata or {}
    headers = get_auth_headers()
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=TIMEOUT)
    return handle_api_error(resp)


def api_match_resume(candidate_json, jd_text):
    url = f"{RESUME_BASE}/resume/match_resume"
    payload = {"candidate": candidate_json, "jd_text": jd_text}
    headers = {"Content-Type": "application/json"}
    headers.update(get_auth_headers())
    resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    return handle_api_error(resp)


def api_score_resume(
    candidate_json=None,
    candidate_id=None,
    jd_text=None,
    role=None,
    profile_weights=None,
):
    url = f"{RESUME_BASE}/resume/score_resume"
    payload = {}
    if candidate_json is not None:
        payload["candidate"] = candidate_json
    if candidate_id is not None:
        payload["candidate_id"] = candidate_id
    if jd_text is not None:
        payload["jd_text"] = jd_text
    if role is not None:
        payload["role"] = role
    if profile_weights is not None:
        payload["profile_weights"] = profile_weights
    headers = {"Content-Type": "application/json"}
    headers.update(get_auth_headers())
    resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    return handle_api_error(resp)


def api_list_docs():
    url = f"{RESUME_BASE}/resume/list_docs"
    headers = get_auth_headers()
    resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    return handle_api_error(resp)


def api_score_batch_upload(zip_bytes, filename="batch.zip", metadata=None):
    url = f"{RESUME_BASE}/resume/score_batch/upload"
    files = {
        "file": (
            filename,
            zip_bytes.getvalue() if hasattr(zip_bytes, "getvalue") else zip_bytes,
            "application/zip",
        )
    }
    data = metadata or {}
    headers = get_auth_headers()
    # increase timeout for uploads
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    return handle_api_error(resp)


def api_score_batch(batch_id=None):
    url = f"{RESUME_BASE}/resume/score_batch"
    headers = get_auth_headers()
    params = {"batch_id": batch_id} if batch_id else {}
    resp = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    return handle_api_error(resp)


def poll_batch_status(batch_id, poll_interval=3, max_wait=300):
    start = time.time()
    while time.time() - start < max_wait:
        resp = api_score_batch(batch_id=batch_id)
        status = resp.get("status") or resp.get("job_status") or resp.get("state")
        # normalize common success indicators
        if status and str(status).lower() in (
            "done",
            "completed",
            "finished",
            "success",
        ):
            return resp
        if status and str(status).lower() in ("failed", "error"):
            raise Exception("Batch job failed: " + json.dumps(resp))
        time.sleep(poll_interval)
    raise TimeoutError("Batch polling timed out")


# ---- UI helper to render score explainability ----
def render_score(score_obj, top_k=None):
    """
    Render score object with Top-K support.
    - If score_obj contains a ranked list of candidates under 'ranked_candidates' or 'candidates',
      show top_k candidates summary and allow selecting one to view details.
    - Otherwise treat as single-candidate score and show per-feature Top-K features (by contribution).
    """
    if not score_obj:
        st.info("No score to display.")
        return

    # If server returned ranked candidates (batch scoring)
    ranked = (
        score_obj.get("ranked_candidates")
        or score_obj.get("candidates")
        or score_obj.get("results")
    )
    if ranked and isinstance(ranked, list):
        k = int(top_k or score_obj.get("top_k", 5))
        st.subheader(f"Top {k} candidates (server ranking)")
        # Build table of top_k
        rows = []
        for c in ranked[:k]:
            # expected fields: candidate_id, final_score or score
            cid = c.get("candidate_id") or c.get("id") or c.get("candidate_id")
            fs = c.get("final_score") or c.get("score") or c.get("combined_score")
            rows.append({"candidate_id": cid, "score": round(fs or 0.0, 4)})
        if rows:
            df = pd.DataFrame(rows).set_index("candidate_id")
            st.table(df)
            # let user pick one to expand details (if available)
            pick = st.selectbox(
                "Select candidate to view details",
                options=list(range(len(ranked[:k]))),
                format_func=lambda i: f"{ranked[i].get('candidate_id', ranked[i].get('id','N/A'))} — {round((ranked[i].get('final_score') or ranked[i].get('score') or 0.0),4)}",
            )
            chosen = ranked[pick]
            st.markdown("#### Details for selected candidate")
            # If backend included full candidate object or per_feature map, show them
            if chosen.get("score_result"):
                render_score(chosen["score_result"], top_k=top_k)
            elif (
                chosen.get("per_feature")
                or chosen.get("features")
                or chosen.get("final_score")
            ):
                # treat chosen as a single-candidate score object
                render_score(chosen, top_k=top_k)
            elif chosen.get("candidate"):
                # if candidate entry contains nested score
                st.json(chosen["candidate"])
            else:
                st.json(chosen)
        return

    # Otherwise single-candidate score
    final_score = (
        score_obj.get("final_score")
        or score_obj.get("score")
        or score_obj.get("combined_score")
    )
    if final_score is None:
        st.subheader("Score response (raw)")
        st.json(score_obj)
        return

    st.subheader(f"Final score: {final_score:.4f}")
    per = score_obj.get("per_feature") or score_obj.get("features") or {}
    if not isinstance(per, dict):
        # sometimes 'features' comes as a list — try convert
        try:
            per = {k: v for k, v in (per or [])}
        except Exception:
            per = {}

    # Build contribution rows and support top_k on features
    rows = []
    for feat, info in per.items():
        rows.append(
            {
                "feature": feat,
                "weight": float(info.get("final_weight", info.get("weight", 0.0))),
                "sim": float(info.get("semantic_similarity", 0.0)),
                "contribution": float(info.get("contribution", 0.0)),
            }
        )

    if not rows:
        st.info("No per-feature information available in the score object.")
        return

    # Apply top_k filtering to features if requested
    k = None
    if top_k:
        try:
            k = int(top_k)
        except Exception:
            k = None
    top_k_features = k or score_obj.get("top_k") or None

    rows_sorted = sorted(rows, key=lambda r: r["contribution"], reverse=True)
    if top_k_features:
        rows_display = rows_sorted[:top_k_features]
        st.caption(
            f"Showing top {top_k_features} contributing features (by contribution)."
        )
    else:
        rows_display = rows_sorted

    df = pd.DataFrame(rows_display).set_index("feature")
    st.table(df)
    st.bar_chart(df["contribution"])

    # Evidence expanders for displayed features
    for feat_row in rows_display:
        feat = feat_row["feature"]
        info = per.get(feat, {})
        with st.expander(
            f"{feat} — sim {info.get('semantic_similarity',0.0):.3f} — contrib {info.get('contribution',0.0):.3f}"
        ):
            ev = info.get("evidence") or info.get("examples") or []
            if not ev:
                st.write("_No evidence available for this feature._")
            else:
                for i, e in enumerate(ev[:30], 1):
                    st.write(f"{i}. {e}")


# ---- Recruiter Dashboard (full) ----
def recruiter_dashboard():
    import uuid  # local import so we don't need to edit top-level imports

    st.title("Recruiter Dashboard")
    st.markdown(
        "### Guided Workflow — 4 steps: (1) Job details → (2) Upload & Process → "
        "(3) Get Top-K → (4) Server docs & Summary"
    )

    if "access_token" not in st.session_state:
        st.info("You are not logged in. Please login via the Login page.")
        return

    # Initialize step state
    if "recruiter_step" not in st.session_state:
        st.session_state["recruiter_step"] = 1
    step = st.session_state["recruiter_step"]

    # Safe navigation helper
    def go_to_step(n):
        st.session_state["recruiter_step"] = n
        try:
            st.experimental_rerun()
        except Exception:
            return

    # ----------------- STEP 1 -----------------
    if step == 1:
        st.header("Step 1 — Job Details")
        jd_text = st.text_area(
            "Job Description",
            height=200,
            key="jd_step1",
            placeholder="Paste the job description or requirements here...",
        )
        kw_text = st.text_input(
            "Required keywords (comma-separated)", placeholder="e.g. Python, NLP, SQL"
        )
        required_keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
        top_k = st.number_input(
            "Default Top-K (will be editable in Step 3)",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )

        # Role selector (keeps recruiter intent explicit)
        role = st.selectbox(
            "Role / Profile (affects scoring priorities)",
            ["developer", "researcher", "analyst"],
            index=0,
        )

        col1, col2 = st.columns([1, 1])
        if col1.button("Next → Step 2"):
            if not jd_text.strip():
                st.warning("Please provide a Job Description before proceeding.")
            else:
                st.session_state["jd_text"] = jd_text
                st.session_state["required_keywords"] = required_keywords
                st.session_state["top_k"] = int(top_k)
                st.session_state["role"] = role
                go_to_step(2)
                return
        col2.write("Tip: Required keywords help prioritize terms during scoring.")

    # ----------------- STEP 2 -----------------
    elif step == 2:
        st.header("Step 2 — Upload & Process Resumes (Batch preferred)")
        st.write(
            "Option A (recommended): Upload multiple resumes and click **Process Batch** to get `doc_ids` from the server."
        )
        st.write(
            "Option B: Upload a single resume and click **Process Single** (uses existing process endpoint)."
        )

        uploaded = st.file_uploader(
            "Select PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            key="step2_upload",
        )
        if uploaded:
            st.write(f"{len(uploaded)} file(s) selected:")
            for f in uploaded:
                st.write("-", f.name)

        # Buttons: Back, Process Batch, Process Single
        b1, b2, b3 = st.columns([1, 1, 1])
        if b1.button("⬅ Back to Step 1"):
            go_to_step(1)
            return

        # Process as batch (multipart files[] + form-data)
        if b2.button("Process Batch (upload files[] + JD)"):
            if not uploaded or len(uploaded) == 0:
                st.warning("Please select at least one PDF file to upload.")
            else:
                jd = st.session_state.get("jd_text", "")
                if not jd.strip():
                    st.warning(
                        "Job Description missing — please go back to Step 1 and enter it."
                    )
                else:
                    # Build files[] payload inline
                    files_payload = []
                    for f in uploaded:
                        content = f.getvalue()
                        files_payload.append(
                            ("files", (f.name, content, "application/pdf"))
                        )

                    data = {
                        "job_description": jd,
                        "top_k": str(st.session_state.get("top_k", 5)),
                        "role": st.session_state.get("role", "developer"),
                    }
                    if st.session_state.get("required_keywords"):
                        data["required_keywords"] = json.dumps(
                            st.session_state.get("required_keywords")
                        )

                    headers = get_auth_headers()
                    try:
                        with st.spinner("Uploading batch to resume service..."):
                            resp = requests.post(
                                f"{RESUME_BASE}/resume/score_batch/upload",
                                headers=headers,
                                files=files_payload,
                                data=data,
                                timeout=120,
                            )

                        # parse response safely
                        try:
                            j = resp.json()
                        except Exception:
                            resp.raise_for_status()
                            j = {}

                        if resp.status_code >= 400:
                            st.error(f"Upload failed: {resp.status_code}")
                            st.json(j)
                        else:
                            # Extract doc_ids from server response that uses "results" array with "doc_id"
                            def extract_doc_ids(resp_obj):
                                out = []
                                if not resp_obj:
                                    return out
                                # Preferred shape: { "results": [ { "doc_id": "..."}, ... ], ... }
                                results = resp_obj.get("results")
                                if isinstance(results, list) and len(results) > 0:
                                    for item in results:
                                        if isinstance(item, dict):
                                            did = (
                                                item.get("doc_id")
                                                or item.get("id")
                                                or item.get("document_id")
                                            )
                                            if did:
                                                out.append(str(did))
                                    return out
                                # Fallback: look for other keys
                                for key in (
                                    "doc_ids",
                                    "documents",
                                    "generated_doc_ids",
                                    "ids",
                                ):
                                    val = resp_obj.get(key)
                                    if val:
                                        if isinstance(val, list):
                                            # list of dicts or scalars
                                            if len(val) > 0 and isinstance(
                                                val[0], dict
                                            ):
                                                for d in val:
                                                    out.append(
                                                        str(
                                                            d.get("doc_id")
                                                            or d.get("id")
                                                            or d.get("document_id")
                                                        )
                                                    )
                                            else:
                                                out = [str(x) for x in val]
                                            return out
                                # if top-level is list of dicts
                                if isinstance(resp_obj, list):
                                    for item in resp_obj:
                                        if isinstance(item, dict):
                                            did = item.get("doc_id") or item.get("id")
                                            if did:
                                                out.append(str(did))
                                    return out
                                return out

                            doc_ids = extract_doc_ids(j)

                            # batch id fallback
                            batch_id = (
                                j.get("batch_id")
                                or j.get("id")
                                or (j.get("data") and j["data"].get("batch_id"))
                                or None
                            )
                            if not batch_id:
                                batch_id = str(uuid.uuid4())[:8]

                            # Persist results
                            st.session_state["last_batch_upload_response"] = j
                            st.session_state["last_batch_id"] = batch_id
                            st.session_state["last_doc_ids"] = doc_ids

                            st.success("Batch uploaded / processed on server.")
                            st.write("Batch ID:", batch_id)
                            st.write("Returned doc_ids count:", len(doc_ids))

                            if len(doc_ids) == 0:
                                st.warning(
                                    "Server returned no doc_ids. We attempted to parse 'results' and other keys."
                                )
                                st.caption("See raw response below for debugging:")
                                st.json(j)
                    except Exception as e:
                        st.error(f"Batch upload failed: {e}")
                        st.info(
                            "If you see a 422 Unprocessable Entity, ensure the server expects 'files' fields and 'job_description' in form-data."
                        )

        # Persisted Proceed button (outside fragile branch)
        if st.session_state.get("last_doc_ids"):
            if st.button("Proceed → Step 3: Get Top-K"):
                go_to_step(3)
                return
        else:
            st.caption(
                "After a successful batch upload you can click 'Proceed → Step 3' (enabled when doc_ids are available)."
            )

        # Process a single selected resume using existing wrapper
        if b3.button("Process Single (use process_resume endpoint)"):
            if not uploaded or len(uploaded) == 0:
                st.warning("Please upload and select a resume first.")
            else:
                sel = uploaded[0]
                try:
                    with st.spinner("Processing resume (single) on server..."):
                        resp = api_process_resume(
                            sel.getvalue(),
                            sel.name,
                            metadata={"source": "recruiter_ui"},
                        )
                    candidate = resp.get("candidate") or resp
                    st.session_state["last_candidate"] = candidate
                    st.success("Single resume processed and loaded.")
                    st.json(candidate)
                    if st.button("Proceed → Step 3: Score this resume"):
                        go_to_step(3)
                        return
                except Exception as e:
                    st.error(f"Single resume processing failed: {e}")

    # ----------------- STEP 3 -----------------
    elif step == 3:
        st.header("Step 3 — Get Top-K / Score")
        st.write(
            "Edit Top-K below if you want to change the number of candidates shown."
        )
        top_k_current = st.number_input(
            "Top-K to return",
            min_value=1,
            max_value=50,
            value=st.session_state.get("top_k", 5),
            key="topk_step3",
        )
        st.session_state["top_k"] = int(top_k_current)

        jd = st.session_state.get("jd_text", "")
        st.text_area("Job Description (readonly)", value=jd, height=140, disabled=True)

        doc_ids = st.session_state.get("last_doc_ids") or []
        candidate = st.session_state.get("last_candidate")

        c1, c2, c3 = st.columns([1, 1, 1])
        if c1.button("⬅ Back to Step 2"):
            go_to_step(2)
            return

        # If we have doc_ids, call score_batch and render a safe table + interactive details
        if doc_ids:
            if c2.button("Get Top-K (call /resume/score_batch)"):
                payload = {
                    "doc_ids": doc_ids,
                    "job_description": jd,
                    "top_k": int(st.session_state.get("top_k", 5)),
                    "role": st.session_state.get("role", "developer"),
                }
                if st.session_state.get("required_keywords"):
                    payload["required_keywords"] = st.session_state.get(
                        "required_keywords"
                    )
                headers = {"Content-Type": "application/json"}
                headers.update(get_auth_headers())
                try:
                    with st.spinner("Requesting top candidates from server..."):
                        resp = requests.post(
                            f"{RESUME_BASE}/resume/score_batch",
                            headers=headers,
                            json=payload,
                            timeout=120,
                        )
                    try:
                        j = resp.json()
                    except Exception:
                        resp.raise_for_status()
                        j = {}

                    if resp.status_code >= 400:
                        st.error(f"Batch scoring failed: {resp.status_code}")
                        st.json(j)
                    else:
                        st.success("Received batch scoring results.")
                        # Normalize results array
                        results = (
                            j.get("results")
                            or j.get("ranked_candidates")
                            or j.get("candidates")
                            or []
                        )
                        if not isinstance(results, list):
                            results = []

                        # Build sanitized rows: scalar candidate_id and score only
                        rows = []
                        for r in results:
                            if not isinstance(r, dict):
                                continue
                            cid = (
                                r.get("candidate_id")
                                or r.get("doc_id")
                                or r.get("id")
                                or r.get("document_id")
                                or "unknown-id"
                            )
                            score_val = (
                                r.get("final_score")
                                or r.get("score")
                                or r.get("combined_score")
                                or 0.0
                            )
                            try:
                                score_val = float(score_val)
                            except Exception:
                                score_val = 0.0
                            rows.append(
                                {"candidate_id": str(cid), "score": score_val, "raw": r}
                            )

                        if len(rows) == 0:
                            st.warning(
                                "No scoring results returned by server. See raw response for debugging."
                            )
                            st.json(j)
                            st.session_state["last_batch_scores"] = j
                        else:
                            # sort by score desc
                            rows_sorted = sorted(
                                rows, key=lambda x: x["score"], reverse=True
                            )
                            k = st.session_state.get("top_k", 5)
                            top_rows = rows_sorted[:k]

                            # Create DataFrame only with scalar columns to avoid Quiver JS errors
                            df = pd.DataFrame(
                                [
                                    {
                                        "candidate_id": r["candidate_id"],
                                        "score": r["score"],
                                    }
                                    for r in top_rows
                                ]
                            ).set_index("candidate_id")
                            st.subheader(
                                f"Top {min(k, len(top_rows))} candidates (server ranking)"
                            )
                            st.table(df)

                            # Build format list for selectbox (use indices + formatted labels)
                            labels = [
                                f"{r['candidate_id']} — {r['score']:.4f}"
                                for r in top_rows
                            ]
                            idx = st.selectbox(
                                "Select candidate to view details",
                                options=list(range(len(labels))),
                                format_func=lambda i: labels[i],
                            )

                            chosen = top_rows[idx]
                            st.markdown("#### Details for selected candidate")
                            st.write("**Candidate (doc_id)**:", chosen["candidate_id"])
                            st.write("**Final score**:", f"{chosen['score']:.4f}")

                            # Show raw JSON for debugging / deeper inspect
                            with st.expander("Show raw candidate payload"):
                                st.json(chosen["raw"])

                            # If backend included nested score_result or per_feature, try rendering it
                            nested_score = (
                                chosen["raw"].get("score_result")
                                or chosen["raw"].get("score")
                                or chosen["raw"].get("score_obj")
                                or chosen["raw"]
                            )
                            try:
                                render_score(
                                    nested_score, top_k=st.session_state.get("top_k", 5)
                                )
                            except Exception:
                                st.info(
                                    "Per-feature explanation not available or failed to render for this candidate."
                                )

                            # Save last_batch_scores and results for Step 4 summary
                            st.session_state["last_batch_scores"] = j

                        # Determine total processed using server 'count' if present (coerce to int), else fallback to length of results
                        total_processed = None
                        if isinstance(j.get("count"), (int, float)):
                            total_processed = int(j.get("count"))
                        elif (
                            isinstance(j.get("count"), str) and j.get("count").isdigit()
                        ):
                            total_processed = int(j.get("count"))
                        else:
                            total_processed = len(results)
                        st.info(
                            f"Total processed: {total_processed} | Top-K shown: {min(st.session_state.get('top_k', 5), len(results))}"
                        )

                except Exception as e:
                    st.error(f"Batch scoring request failed: {e}")
                    st.info(
                        "If server returns 422, confirm payload contains doc_ids (array), job_description (string), and top_k (int)."
                    )

        # Single-candidate scoring path (if user processed single resume earlier)
        elif candidate:
            doc_id = (
                candidate.get("doc_id")
                or candidate.get("candidate_id")
                or candidate.get("id")
            )
            if doc_id:
                if c2.button("Score this single doc (call /resume/score_resume)"):
                    try:
                        with st.spinner("Scoring single document on server..."):
                            resp = api_score_resume(
                                doc_id=doc_id,
                                job_description=jd,
                                required_keywords=st.session_state.get(
                                    "required_keywords", []
                                ),
                                top_k=st.session_state.get("top_k", 5),
                                role=st.session_state.get("role", "developer"),
                            )
                        st.session_state["last_score"] = resp
                        render_score(resp, top_k=st.session_state.get("top_k", 5))
                        st.success("Single scoring completed.")
                    except Exception as e:
                        st.error(f"Single scoring failed: {e}")
                        st.info(
                            "If scoring endpoint expects 'doc_id' in JSON, ensure wrapper sends that shape."
                        )
            else:
                st.info(
                    "Processed candidate does not expose a doc_id. You can re-process via batch upload to get doc_ids, or pass candidate JSON to the backend if supported."
                )

        # allow skip to Step 4
        if c3.button("Proceed → Step 4 (Summary & Server Docs)"):
            go_to_step(4)
            return

    # ----------------- STEP 4 (PATCHEd: no remote /resume/list_docs call) -----------------
    elif step == 4:
        st.header("Step 4 — Server Docs & Summary")
        st.write(
            "Summary of processed resumes from your last batch operation. This Step uses the stored batch results (from Step 3) for totals and downloads. The UI no longer depends on a /resume/list_docs endpoint."
        )

        a1, a2 = st.columns([1, 1])
        if a1.button("⬅ Back to Step 3"):
            go_to_step(3)
            return

        # "Refresh server documents" removed remote call. Instead show guidance or use existing session data.
        if a2.button("Load last batch summary (from session)"):
            st.info(
                "This loads the last batch results saved during your session (from Step 3). If you need a server-wide listing, add a /resume/list_docs endpoint to the backend."
            )
            # no network call; we just re-display stored data below.

        # Primary source of truth: last_batch_scores (set when calling /resume/score_batch)
        last_batch_scores = st.session_state.get("last_batch_scores")
        last_batch_upload_resp = st.session_state.get("last_batch_upload_response")
        last_doc_ids = st.session_state.get("last_doc_ids") or []

        # Show upload-time summary if available
        if last_batch_upload_resp:
            st.subheader("Upload response (raw)")
            st.json(last_batch_upload_resp)

        # Show scoring results summary if available
        if last_batch_scores:
            # Prefer server 'count' if present
            total_processed_val = (
                last_batch_scores.get("count")
                or last_batch_scores.get("total_processed")
                or len(last_batch_scores.get("results", []))
            )
            try:
                total_processed_val = int(total_processed_val)
            except Exception:
                pass
            st.metric("Total processed (last batch)", total_processed_val)

            results = last_batch_scores.get("results") or []
            st.metric("Candidates returned by scoring", len(results))

            # Provide download
            st.download_button(
                "Download last batch results (JSON)",
                json.dumps(last_batch_scores, indent=2),
                file_name="batch_results.json",
            )

            # Show a small preview of top results if present
            if isinstance(results, list) and len(results) > 0:
                preview_rows = []
                for r in results[: min(10, len(results))]:
                    cid = (
                        r.get("candidate_id")
                        or r.get("doc_id")
                        or r.get("id")
                        or "unknown-id"
                    )
                    score_val = r.get("final_score") or r.get("score") or 0.0
                    try:
                        score_val = float(score_val)
                    except Exception:
                        score_val = 0.0
                    preview_rows.append({"candidate_id": cid, "score": score_val})
                df_preview = pd.DataFrame(preview_rows).set_index("candidate_id")
                st.subheader("Top results preview")
                st.table(df_preview)

        else:
            st.info(
                "No batch scoring results found in session. Run Step 3 to obtain Top-K results first."
            )

        # Server docs listing is only shown if present in session (we don't call the backend)
        server_docs = st.session_state.get("server_docs") or []
        if server_docs:
            st.subheader("Server documents (session cache preview)")
            try:
                df = pd.DataFrame(server_docs)
                st.dataframe(df.head(100))
            except Exception:
                st.write(server_docs)
        else:
            st.caption(
                "Server document listing is not available in session. To enable a full server-side listing, implement GET /resume/list_docs on the backend and re-enable the UI call."
            )

        # Final action: start new job (clears state)
        if st.button("Start a new job (clear and go to Step 1)"):
            keys_to_clear = [
                "last_batch_upload_response",
                "last_batch_id",
                "last_doc_ids",
                "last_batch_scores",
                "last_score",
                "last_candidate",
                "server_docs",
            ]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state["recruiter_step"] = 1
            go_to_step(1)
            return


# ---- Existing pages: Signup, Login, Admin (kept mostly as-is, now point to AUTH_BASE) ----
def signup_page():
    st.title("HireSense - Recruiter Signup")
    with st.form("signup_form"):
        name = st.text_input("Full name")
        recruiter_role = st.selectbox(
            "Role", ["HR", "Hiring Manager", "Tech Lead", "Sourcer"]
        )
        business = st.text_input("Business name")
        website = st.text_input("Website URL")
        employees = st.selectbox(
            "No. of employees", ["0-25", "25-50", "50-100", "100-300", "300+"]
        )
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign up")
    if submitted:
        payload = {
            "recruiter_name": name,
            "recruiter_role": recruiter_role,
            "business_name": business,
            "website_url": website or None,
            "no_of_employees": employees,
            "email": email,
            "phone": phone,
            "password": password,
        }
        resp = requests.post(f"{AUTH_BASE}/auth/signup", json=payload)
        if resp.status_code in (200, 202):
            st.success(
                "Signup request received. Our team will verify your data within 2 working days."
            )
        else:
            try:
                st.error(resp.json().get("detail", "Error"))
            except Exception:
                st.error("Signup failed")


def login_page():
    st.title("HireSense - Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        resp = requests.post(
            f"{AUTH_BASE}/auth/login", json={"email": email, "password": password}
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state["access_token"] = data["access_token"]
            st.success("Login successful")
        else:
            try:
                st.error(resp.json().get("detail", "Login failed"))
            except Exception:
                st.error("Login failed")


def admin_dashboard():
    st.title("Admin Dashboard")
    token = st.session_state.get("access_token")
    if not token:
        st.warning("Please login as admin first")
        return
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(f"{AUTH_BASE}/admin/pending-signups", headers=headers)
        if resp.status_code == 200:
            pending = resp.json()
            st.write(f"Pending signups: {len(pending)}")
            for u in pending:
                st.write(
                    f"ID: {u['id']} | Name: {u['full_name']} | Email: {u['email']}"
                )
                cols = st.columns(3)
                if cols[0].button(f"Approve {u['id']}"):
                    r = requests.post(
                        f"{AUTH_BASE}/admin/approve/{u['id']}", headers=headers
                    )
                    st.write(r.json())
                if cols[1].button(f"Reject {u['id']}"):
                    r = requests.post(
                        f"{AUTH_BASE}/admin/reject/{u['id']}",
                        headers=headers,
                        json={"reason": "Rejected via admin UI"},
                    )
                    st.write(r.json())
        else:
            st.error(
                "Failed to fetch pending signups. Ensure your token is admin token."
            )
    except Exception as e:
        st.error("Failed to call admin endpoint: " + str(e))


# ---- Simple routing ----
st.sidebar.title("HireSense")
page = st.sidebar.selectbox(
    "Go to", ["Signup", "Login", "Recruiter Dashboard", "Admin Dashboard"]
)
if page == "Signup":
    signup_page()
elif page == "Login":
    login_page()
elif page == "Recruiter Dashboard":
    recruiter_dashboard()
elif page == "Admin Dashboard":
    admin_dashboard()
