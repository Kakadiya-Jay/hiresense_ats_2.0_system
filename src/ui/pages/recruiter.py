# src/pages/recruiter.py
"""
Recruiter dashboard page — patched to keep a frontend-only mapping (Streamlit session_state)
between doc_id and filename/final_score returned by the upload endpoint. This avoids any server DB.
See: session_state["last_upload_mapping"] which stores { doc_id: {"filename":..., "final_score":...} }
"""

from src.ui.context import st, safe_rerun, get_auth_headers, json
from src.ui.ui_helpers.navigation import render_top_right_user
from src.api.ui_integration.resume_api import (
    api_process_resume,
    api_score_resume,
    api_score_batch_upload,
    api_score_batch,
)
from src.ui.score_renderer import render_score
import uuid
import pandas as pd
import traceback

# Ensure session state keys used by this page exist
if "recruiter_step" not in st.session_state:
    st.session_state["recruiter_step"] = 1
if "last_upload_response" not in st.session_state:
    st.session_state["last_upload_response"] = None
if "last_upload_mapping" not in st.session_state:
    st.session_state["last_upload_mapping"] = {}
# Keep old compatibility keys too
if "last_batch_upload_response" not in st.session_state:
    st.session_state["last_batch_upload_response"] = None
if "last_doc_ids" not in st.session_state:
    st.session_state["last_doc_ids"] = []
if "last_batch_id" not in st.session_state:
    st.session_state["last_batch_id"] = None


def recruiter_dashboard():
    st.title("Recruiter Dashboard")
    st.markdown(
        "### Guided Workflow — 4 steps: (1) Job details → (2) Upload & Process → "
        "(3) Get Top-K → (4) Server docs & Summary"
    )

    if "access_token" not in st.session_state:
        st.info("You are not logged in. Please login via the Login page.")
        return

    step = st.session_state["recruiter_step"]

    def go_to_step(n):
        st.session_state["recruiter_step"] = n
        # CHANGED: use safe_rerun provided by context (version-compatible)
        try:
            safe_rerun()
        except Exception:
            # safe fallback: write a small message (should rarely be needed)
            st.warning("Navigation requested; please interact to refresh the UI.")

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
        role_select = st.selectbox(
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
                st.session_state["role"] = role_select
                go_to_step(2)
                return
        col2.write("Tip: Required keywords help prioritize terms during scoring.")

    # ----------------- STEP 2 -----------------
    elif step == 2:
        st.header("Step 2 — Upload & Process Resumes (Batch preferred)")
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

        b1, b2, b3 = st.columns([1, 1, 1])
        if b1.button("⬅ Back to Step 1"):
            go_to_step(1)
            return

        # Upload & process button (robust — do NOT return after safe_rerun)
        if b2.button("Upload & Process (call /resume/score_batch/upload)"):
            if not uploaded:
                st.warning("Please select one or more PDF files to upload.")
            else:
                files_payload = []
                for f in uploaded:
                    try:
                        content = f.read()
                    except Exception:
                        f.seek(0)
                        content = f.read()
                    files_payload.append(
                        ("files", (f.name, content, "application/pdf"))
                    )

                data = {
                    "job_description": st.session_state.get("jd_text", ""),
                    "required_keywords": ",".join(
                        st.session_state.get("required_keywords", [])
                    ),
                    "top_k": str(st.session_state.get("top_k", 5)),
                }

                with st.spinner("Uploading and processing files..."):
                    try:
                        upload_resp = api_score_batch_upload(files_payload, data)

                        # Normalize response (requests.Response -> dict)
                        if hasattr(upload_resp, "json") and not isinstance(
                            upload_resp, dict
                        ):
                            try:
                                upload_resp = upload_resp.json()
                            except Exception:
                                try:
                                    upload_resp = json.loads(upload_resp.text)
                                except Exception:
                                    raise RuntimeError(
                                        "Upload endpoint returned non-json response"
                                    )

                        # Persist session keys (both new + legacy keys)
                        st.session_state["last_upload_response"] = upload_resp
                        st.session_state["last_batch_upload_response"] = upload_resp

                        # Extract doc_ids robustly
                        doc_ids = upload_resp.get("doc_ids") or []
                        if not doc_ids:
                            results_obj = (
                                upload_resp.get("results")
                                or upload_resp.get("files")
                                or upload_resp.get("documents")
                                or []
                            )
                            for item in results_obj:
                                if isinstance(item, dict):
                                    did = (
                                        item.get("doc_id")
                                        or item.get("id")
                                        or item.get("document_id")
                                    )
                                    if did:
                                        doc_ids.append(str(did))
                                else:
                                    doc_ids.append(str(item))
                        st.session_state["last_doc_ids"] = doc_ids

                        # store batch id (if provided)
                        st.session_state["last_batch_id"] = (
                            upload_resp.get("batch_id")
                            or upload_resp.get("id")
                            or st.session_state.get("last_batch_id")
                        )

                        # Build mapping for UI
                        mapping = st.session_state.get("last_upload_mapping", {}) or {}
                        try:
                            file_list = (
                                upload_resp.get("results")
                                or upload_resp.get("files")
                                or []
                            )
                            for item in file_list:
                                if not isinstance(item, dict):
                                    continue
                                did = (
                                    item.get("doc_id")
                                    or item.get("docid")
                                    or item.get("id")
                                )
                                fname = item.get("filename") or item.get("file_name")
                                final_score = item.get("final_score")
                                if did:
                                    mapping[did] = {
                                        "filename": fname,
                                        "final_score": final_score,
                                    }
                            st.session_state["last_upload_mapping"] = mapping
                        except Exception as map_exc:
                            st.warning(f"Warning: mapping build failed: {map_exc}")

                        st.success("Upload + processing completed.")
                        st.write("Batch ID:", st.session_state.get("last_batch_id"))
                        st.write(
                            "Returned doc_ids count:",
                            len(st.session_state.get("last_doc_ids", [])),
                        )

                        # Attempt to rerun the app to immediately reflect navigation
                        try:
                            safe_rerun()
                            # DO NOT return here — if safe_rerun is a no-op, we still want to continue
                        except Exception:
                            # safe_rerun should not crash; ignore
                            pass

                    except Exception as e:
                        import traceback

                        tb = traceback.format_exc()
                        st.exception(f"Upload failed: {e}")
                        st.text(tb)

        # Keep old gating: enable Proceed → Step 3 if last_doc_ids present
        if st.button("Proceed → Step 3: Get Top-K"):
            doc_ids = st.session_state.get("last_doc_ids") or []
            # debug help: print count to UI (you can remove in prod)
            try:
                st.write(f"Debug: last_doc_ids count = {len(doc_ids)}")
            except Exception:
                pass

            if not doc_ids:
                st.warning(
                    "No doc_ids found from last upload — please upload resumes first or press 'Upload & Process' again."
                )
            else:
                go_to_step(3)
                return

        # Friendly caption (keeps the UX hint visible)
        st.caption(
            "Tip: After upload finishes the app populates doc_ids automatically. If you do not see results, try clicking this button to proceed."
        )

        if b3.button("Process Single (use process_resume endpoint)"):
            if not uploaded or len(uploaded) == 0:
                st.warning("Please upload and select a resume first.")
            else:
                sel = uploaded[0]
                try:
                    with st.spinner("Processing resume (single) on server..."):
                        resp = api_process_resume(
                            sel.read(), sel.name, metadata={"source": "recruiter_ui"}
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
    # ----------------- STEP 3 -----------------
    elif step == 3:
        st.header("Step 3 — Get Top-K / Score")
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

        # Try to reuse cached scores if present
        cached_scores = st.session_state.get("last_batch_scores")
        scores_payload = None
        used_cached = False
        if cached_scores:
            scores_payload = cached_scores
            used_cached = True

        # If user asks for new Top-K, fetch and overwrite cache
        if c2.button("Get Top-K (call /resume/score_batch)"):
            payload = {
                "doc_ids": doc_ids,
                "job_description": jd,
                "top_k": int(st.session_state.get("top_k", 5)),
                "role": st.session_state.get("role", "developer"),
            }
            rk = st.session_state.get("required_keywords")
            if rk:
                if isinstance(rk, list):
                    payload["required_keywords"] = ", ".join(
                        [
                            str(x).strip()
                            for x in rk
                            if x is not None and str(x).strip() != ""
                        ]
                    )
                else:
                    payload["required_keywords"] = str(rk)
            try:
                with st.spinner("Requesting top candidates from server..."):
                    j = api_score_batch(payload)
                # cache for re-runs / select changes
                st.session_state["last_batch_scores"] = j
                scores_payload = j
                used_cached = True
            except Exception as e:
                st.error(f"Batch scoring request failed: {e}")
                st.info(
                    "If server returns 422, confirm payload contains doc_ids (array), job_description (string), and top_k (int)."
                )
                scores_payload = None
                used_cached = False

        # If we have scores from either cache or fresh request, build UI
        if scores_payload and used_cached:
            results = (
                scores_payload.get("results")
                or scores_payload.get("ranked_candidates")
                or scores_payload.get("candidates")
                or []
            )
            if not isinstance(results, list):
                results = []

            # Build rows using mapping if present
            rows = []
            mapping = st.session_state.get("last_upload_mapping", {}) or {}
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
                fname = (
                    mapping.get(cid, {}).get("filename")
                    if mapping.get(cid)
                    else (r.get("file_name") or r.get("filename"))
                )
                score_val = (
                    mapping.get(cid, {}).get("final_score")
                    if mapping.get(cid)
                    and mapping.get(cid).get("final_score") is not None
                    else (r.get("final_score") or r.get("score") or 0.0)
                )
                try:
                    score_val = float(score_val)
                except Exception:
                    pass
                rows.append(
                    {
                        "candidate_id": str(cid),
                        "score": score_val,
                        "file_name": fname,
                        "raw": r,
                    }
                )

            if len(rows) == 0:
                st.warning(
                    "No scoring results returned by server. See raw response for debugging."
                )
                st.json(scores_payload)
                st.session_state["last_batch_scores"] = scores_payload
            else:
                rows_sorted = sorted(
                    rows,
                    key=lambda x: (x["score"] is not None, x["score"]),
                    reverse=True,
                )
                k = st.session_state.get("top_k", 5)
                top_rows = rows_sorted[:k]

                # Table
                df = pd.DataFrame(
                    [
                        {
                            "candidate_id": r["candidate_id"],
                            "file_name": r.get("file_name"),
                            "score": r["score"],
                        }
                        for r in top_rows
                    ]
                ).set_index("candidate_id")
                st.subheader(f"Top {min(k, len(top_rows))} candidates (server ranking)")
                st.table(df)

                # Labels and persistent selectbox
                labels = [
                    f"{(r.get('file_name') or 'unknown')} — {r['candidate_id']} — {r['score']:.4f}"
                    for r in top_rows
                ]
                selection_key = "recruiter_topk_select_idx"
                default_index = st.session_state.get(selection_key, 0)
                # use stable key for selectbox so its state is preserved across reruns
                idx = st.selectbox(
                    "Select candidate to view details",
                    options=list(range(len(labels))),
                    index=default_index,
                    format_func=lambda i: labels[i],
                    key="select_candidate_dropdown",
                )
                # persist selected index for future reruns
                st.session_state[selection_key] = idx

                chosen = top_rows[idx]
                st.markdown("#### Details for selected candidate")
                st.write("**Candidate (doc_id)**:", chosen["candidate_id"])
                st.write("**Final score**:", f"{chosen['score']:.4f}")
                st.write("**File name**:", chosen.get("file_name") or "—")

                # Download CV button (see explanation below for backend hookup)
                # Example: clickable link to download endpoint (opens new tab)
                API_BASE = st.session_state.get("api_base_url", "")
                if API_BASE:
                    doc_id_for_download = chosen["candidate_id"]
                    fname_for_download = (
                        chosen.get("file_name") or f"{doc_id_for_download}.pdf"
                    )
                    download_url = (
                        f"{API_BASE.rstrip('/')}/resume/download/{doc_id_for_download}"
                    )
                    st.markdown(
                        f"<a href='{download_url}' target='_blank'><button>Open / Download CV</button></a>",
                        unsafe_allow_html=True,
                    )

                with st.expander("Show raw candidate payload"):
                    st.json(chosen["raw"])

                nested_score = (
                    chosen["raw"].get("score_result")
                    or chosen["raw"].get("score")
                    or chosen["raw"].get("score_obj")
                    or chosen["raw"]
                )
                try:
                    render_score(nested_score, top_k=st.session_state.get("top_k", 5))
                except Exception:
                    st.info(
                        "Per-feature explanation not available or failed to render for this candidate."
                    )

                # cache last batch scores (reinforce)
                st.session_state["last_batch_scores"] = scores_payload

                # Counts: uploaded vs processed/returned by scoring
                uploaded_count = len(st.session_state.get("last_doc_ids", []))
                # compute scored_count from payload
                if isinstance(scores_payload.get("count"), (int, float)):
                    scored_count = int(scores_payload.get("count"))
                elif (
                    isinstance(scores_payload.get("count"), str)
                    and scores_payload.get("count").isdigit()
                ):
                    scored_count = int(scores_payload.get("count"))
                else:
                    scored_count = len(results)
                st.info(
                    f"Uploaded (sent): {uploaded_count} | Processed/returned by scoring: {scored_count} | Top-K shown: {min(k, len(results))}"
                )

        elif candidate:
            # Single processed candidate (from Step 2 "Process Single")
            doc_id = (
                candidate.get("doc_id")
                or candidate.get("candidate_id")
                or candidate.get("id")
            )
            if doc_id:
                if c2.button("Score this single doc (call /resume/score_resume)"):
                    try:
                        resp = api_score_resume(
                            {
                                "doc_id": doc_id,
                                "job_description": jd,
                                "required_keywords": st.session_state.get(
                                    "required_keywords", []
                                ),
                                "top_k": st.session_state.get("top_k", 5),
                                "role": st.session_state.get("role", "developer"),
                            }
                        )
                        st.session_state["last_score"] = resp
                        render_score(resp, top_k=st.session_state.get("top_k", 5))
                        st.success("Single scoring completed.")
                    except Exception as e:
                        st.error(f"Single scoring failed: {e}")
            else:
                st.info(
                    "Processed candidate does not expose a doc_id. You can re-process via batch upload to get doc_ids, or pass candidate JSON to the backend if supported."
                )
        if c3.button("Proceed → Step 4 (Summary & Server Docs)"):
            go_to_step(4)
            return

    # ----------------- STEP 4 -----------------
    elif step == 4:
        st.header("Step 4 — Server Docs & Summary")
        st.write(
            "Summary of processed resumes from last batch operation. This Step uses the stored batch results (from Step 3) for totals and downloads. The UI no longer depends on a /resume/list_docs endpoint."
        )

        a1, a2 = st.columns([1, 1])
        if a1.button("⬅ Back to Step 3"):
            go_to_step(3)
            return
        if a2.button("Load last batch summary (from session)"):
            st.info(
                "This loads the last batch results saved during session (from Step 3). If you need a server-wide listing, add a /resume/list_docs endpoint to the backend."
            )

        last_batch_scores = st.session_state.get("last_batch_scores")
        last_batch_upload_resp = st.session_state.get("last_batch_upload_response")
        last_doc_ids = st.session_state.get("last_doc_ids") or []

        if last_batch_upload_resp:
            st.subheader("Upload response (raw)")
            st.json(last_batch_upload_resp)

        if last_batch_scores:
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
            st.download_button(
                "Download last batch results (JSON)",
                json.dumps(last_batch_scores, indent=2),
                file_name="batch_results.json",
            )
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

        if st.button("Start a new job (clear and go to Step 1)"):
            keys_to_clear = [
                "last_batch_upload_response",
                "last_batch_id",
                "last_doc_ids",
                "last_batch_scores",
                "last_score",
                "last_candidate",
                "server_docs",
                "last_upload_response",
                "last_upload_mapping",
            ]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state["recruiter_step"] = 1
            go_to_step(1)
            return

    # Render top-right user info (unchanged)
    try:
        render_top_right_user()
    except Exception:
        pass
