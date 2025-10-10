# src/pages/recruiter.py
"""
Recruiter dashboard page extracted from the large file.
This keeps the same flow: 4 steps (JD -> Upload -> Get Top-K -> Summary).
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


def recruiter_dashboard():
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

    def go_to_step(n):
        st.session_state["recruiter_step"] = n
        safe_rerun()

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
                        rk_val = st.session_state.get("required_keywords")
                        if isinstance(rk_val, list):
                            data["required_keywords"] = ", ".join([str(x).strip() for x in rk_val if x and str(x).strip() != ""])
                        else:
                            data["required_keywords"] = str(rk_val)
                    try:
                        with st.spinner("Uploading batch to resume service..."):
                            j = api_score_batch_upload(files_payload, data)
                        # normalize response
                        results = (
                            j.get("results")
                            or j.get("documents")
                            or j.get("doc_ids")
                            or j.get("ids")
                            or []
                        )
                        doc_ids = []
                        if isinstance(results, list):
                            for item in results:
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
                        batch_id = (
                            j.get("batch_id")
                            or j.get("id")
                            or (j.get("data") and j["data"].get("batch_id"))
                            or str(uuid.uuid4())[:8]
                        )
                        st.session_state["last_batch_upload_response"] = j
                        st.session_state["last_batch_id"] = batch_id
                        st.session_state["last_doc_ids"] = doc_ids
                        st.success("Batch uploaded / processed on server.")
                        st.write("Batch ID:", batch_id)
                        st.write("Returned doc_ids count:", len(doc_ids))
                    except Exception as e:
                        st.error(f"Batch upload failed: {e}")
                        st.info(
                            "If you see a 422 Unprocessable Entity, ensure the server expects 'files' fields and 'job_description' in form-data."
                        )

        if st.session_state.get("last_doc_ids"):
            if st.button("Proceed → Step 3: Get Top-K"):
                go_to_step(3)
                return
        else:
            st.caption(
                "After a successful batch upload you can click 'Proceed → Step 3' (enabled when doc_ids are available)."
            )

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

        if doc_ids:
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
                    results = (
                        j.get("results")
                        or j.get("ranked_candidates")
                        or j.get("candidates")
                        or []
                    )
                    if not isinstance(results, list):
                        results = []
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
                        rows_sorted = sorted(
                            rows, key=lambda x: x["score"], reverse=True
                        )
                        k = st.session_state.get("top_k", 5)
                        top_rows = rows_sorted[:k]
                        df = pd.DataFrame(
                            [
                                {"candidate_id": r["candidate_id"], "score": r["score"]}
                                for r in top_rows
                            ]
                        ).set_index("candidate_id")
                        st.subheader(
                            f"Top {min(k, len(top_rows))} candidates (server ranking)"
                        )
                        st.table(df)
                        labels = [
                            f"{r['candidate_id']} — {r['score']:.4f}" for r in top_rows
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
                        with st.expander("Show raw candidate payload"):
                            st.json(chosen["raw"])
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
                        st.session_state["last_batch_scores"] = j
                    total_processed = None
                    if isinstance(j.get("count"), (int, float)):
                        total_processed = int(j.get("count"))
                    elif isinstance(j.get("count"), str) and j.get("count").isdigit():
                        total_processed = int(j.get("count"))
                    else:
                        total_processed = len(results)
                    st.info(
                        f"Total processed: {total_processed} | Top-K shown: {min(st.session_state.get('top_k',5), len(results))}"
                    )
                except Exception as e:
                    st.error(f"Batch scoring request failed: {e}")
                    st.info(
                        "If server returns 422, confirm payload contains doc_ids (array), job_description (string), and top_k (int)."
                    )
        elif candidate:
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
            "Summary of processed resumes from your last batch operation. This Step uses the stored batch results (from Step 3) for totals and downloads. The UI no longer depends on a /resume/list_docs endpoint."
        )
        a1, a2 = st.columns([1, 1])
        if a1.button("⬅ Back to Step 3"):
            go_to_step(3)
            return
        if a2.button("Load last batch summary (from session)"):
            st.info(
                "This loads the last batch results saved during your session (from Step 3). If you need a server-wide listing, add a /resume/list_docs endpoint to the backend."
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
            ]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state["recruiter_step"] = 1
            go_to_step(1)
            return
