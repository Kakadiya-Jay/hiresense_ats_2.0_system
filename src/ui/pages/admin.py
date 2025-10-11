"""
Admin dashboard page for Streamlit app.
Features:
- Equal-sized metric cards
- Paginated tables (10 rows per page)
- Action buttons placed after paginated rows (Activate/Deactivate | Approve/Reject)
- Defensive column selection (avoids KeyError)
- Uses src.ui.components for API calls and auth helpers
"""

from typing import List, Dict, Any
import math
import pandas as pd
from src.ui.context import st, safe_rerun, get_auth_headers
from src.ui import components


# ----------------- Styling helpers -----------------
def _card_html(title: str, value: str, active: bool = False) -> str:
    bg = "#617cff" if active else "#bfc8ff"
    text = "#fff" if active else "#111"
    hover = "#506dff" if active else "#aebaff"
    return f"""
    <div style="
        width:100%;
        height:180px;
        border-radius:16px;
        background:{bg};
        color:{text};
        text-align:center;
        box-shadow:0 6px 18px rgba(0,0,0,0.12);
        transition:all 0.3s ease;
        display:flex;
        flex-direction:column;
        justify-content:center;
        cursor:pointer;
    ">
        <div style="font-size:16px;opacity:0.95;">{title}</div>
        <div style="font-size:40px;font-weight:700;margin-top:10px;">{value}</div>
    </div>
    """


# ----------------- Paginated table renderer -----------------
def _render_paginated_table(
    df: pd.DataFrame, table_type: str = "users", page_key: str = "page_num"
):
    """
    Renders a paginated DataFrame and inline action buttons that appear after the table.
    - table_type: "users" or "pending"
    - page_key: unique per table to store page number in session_state
    """

    # ensure page key exists
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    page_size = 10
    total_rows = len(df)
    total_pages = max(1, math.ceil(total_rows / page_size))
    current_page = int(st.session_state.get(page_key, 1))

    # Pagination controls (unique keys)
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Prev", key=f"{page_key}_prev", disabled=current_page == 1):
            st.session_state[page_key] = max(1, current_page - 1)
            safe_rerun()
    with col_info:
        st.markdown(
            f"<p style='text-align:center;'>Page {current_page} of {total_pages}</p>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button(
            "Next ➡️", key=f"{page_key}_next", disabled=current_page == total_pages
        ):
            st.session_state[page_key] = min(total_pages, current_page + 1)
            safe_rerun()

    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    # show the page of the dataframe using modern param
    st.dataframe(page_df, width="stretch", height=360)

    # Actions: render one action row per row shown on the page
    st.write("")  # small spacer
    st.write("### Actions")
    for local_idx, row in page_df.iterrows():
        # unique row id (try numeric ID first)
        raw_id = row.get("ID") or row.get("id") or None
        # use string id for key-safety
        row_id_str = (
            str(raw_id) if raw_id is not None else f"{current_page}_{local_idx}"
        )

        cols = st.columns([4, 1])
        with cols[0]:
            display_name = (
                row.get("Full name")
                or row.get("full_name")
                or row.get("Email")
                or row_id_str
            )
            display_email = row.get("Email") or row.get("email") or ""
            display_role = row.get("Role") or row.get("role") or ""
            st.markdown(f"**{display_name}** | {display_email} | role: {display_role}")

        with cols[1]:
            # USERS table actions
            if table_type == "users":
                # 'Active' column may be bool or string; normalize to bool
                active_val = (
                    row.get("Active")
                    if "Active" in row
                    else row.get("is_active", False)
                )
                try:
                    is_active = bool(active_val)
                except Exception:
                    is_active = False

                if is_active:
                    btn_key = f"deact_{row_id_str}_{page_key}"
                    if st.button("Deactivate", key=btn_key):
                        try:
                            ok = components.toggle_user_active(
                                int(raw_id), make_active=False
                            )
                        except Exception:
                            ok = False
                        if ok:
                            st.success("User deactivated.")
                            safe_rerun()
                        else:
                            st.error("Failed to deactivate user.")
                else:
                    btn_key = f"act_{row_id_str}_{page_key}"
                    if st.button("Activate", key=btn_key):
                        try:
                            ok = components.toggle_user_active(
                                int(raw_id), make_active=True
                            )
                        except Exception:
                            ok = False
                        if ok:
                            st.success("User activated.")
                            safe_rerun()
                        else:
                            st.error("Failed to activate user.")

            # PENDING table actions
            elif table_type == "pending":
                key_approve = f"approve_{row_id_str}_{page_key}"
                key_reject = f"reject_{row_id_str}_{page_key}"
                subc = st.columns([1, 1])
                with subc[0]:
                    if st.button("Allow", key=key_approve):
                        try:
                            ok = components.admin_approve_user(int(raw_id))
                        except Exception:
                            ok = False
                        if ok:
                            st.success("Approved.")
                            safe_rerun()
                        else:
                            st.error("Approve failed.")
                with subc[1]:
                    if st.button("Reject", key=key_reject):
                        try:
                            ok = components.admin_reject_user(int(raw_id))
                        except Exception:
                            ok = False
                        if ok:
                            st.success("Rejected.")
                            safe_rerun()
                        else:
                            st.error("Reject failed.")


# ----------------- Admin page rendering -----------------
def admin_dashboard():
    st.title("Admin Dashboard")

    # Load current user into session (helpers inside components)
    try:
        components.ensure_me_loaded()
    except Exception:
        # ensure_me_loaded is nice-to-have; continue, but user must be in session_state
        pass

    user = st.session_state.get("current_user")
    if not user:
        st.warning("Please login as admin first.")
        return

    if user.get("role") != "admin":
        st.error("Access denied — admin users only.")
        return

    # fetch data
    try:
        users = components.fetch_admin_users() or []
    except Exception:
        users = []

    try:
        pending = components.fetch_pending_signups() or []
    except Exception:
        pending = []

    resume_count = 0  # placeholder, replace with resume API call if available

    total_users = len(users)
    total_pending = len(pending)
    total_resumes = resume_count

    # Cards row
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Total Number of users", key="card_users"):
            st.session_state["admin_active_card"] = "users"
        st.markdown(
            _card_html(
                "Total Number of users",
                str(total_users),
                active=(st.session_state.get("admin_active_card") == "users"),
            ),
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Total Number of resume processed", key="card_resumes"):
            st.session_state["admin_active_card"] = "resumes"
        st.markdown(
            _card_html(
                "Total Number of resume processed",
                str(total_resumes),
                active=(st.session_state.get("admin_active_card") == "resumes"),
            ),
            unsafe_allow_html=True,
        )
    with c3:
        if st.button("Total Number of pending approvals", key="card_pending"):
            st.session_state["admin_active_card"] = "pending"
        st.markdown(
            _card_html(
                "Total Number of pending approvals",
                str(total_pending),
                active=(st.session_state.get("admin_active_card") == "pending"),
            ),
            unsafe_allow_html=True,
        )

    st.write("---")

    # Determine which card is active and show respective content
    active_card = st.session_state.get("admin_active_card")

    if active_card == "users":
        # defensive column selection for users
        if not users:
            st.info("No users found.")
            return
        df = pd.DataFrame(users)
        wanted = [
            "id",
            "full_name",
            "email",
            "role",
            "recruiter_role",
            "business_name",
            "website_url",
            "phone",
            "is_active",
            "created_at",
        ]
        cols = [c for c in wanted if c in df.columns]
        if not cols:
            st.info("No usable user columns returned from API.")
            return
        df = df[cols].copy().fillna("—")
        # rename present columns consistently
        rename_map = {}
        for c in cols:
            if c == "id":
                rename_map[c] = "ID"
            elif c == "full_name":
                rename_map[c] = "Full name"
            elif c == "email":
                rename_map[c] = "Email"
            elif c == "role":
                rename_map[c] = "Role"
            elif c == "recruiter_role":
                rename_map[c] = "Recruiter role"
            elif c == "business_name":
                rename_map[c] = "Business name"
            elif c == "website_url":
                rename_map[c] = "Website"
            elif c == "phone":
                rename_map[c] = "Phone"
            elif c == "is_active":
                rename_map[c] = "Active"
            elif c == "created_at":
                rename_map[c] = "Created at"
        df = df.rename(columns=rename_map)
        # render with pagination and actions
        _render_paginated_table(df, table_type="users", page_key="user_page")

    elif active_card == "resumes":
        st.info("Resume analytics feature coming soon!")

    elif active_card == "pending":
        if not pending:
            st.info("No pending signups.")
            return
        df = pd.DataFrame(pending)
        wanted = ["id", "full_name", "email", "role", "status", "created_at"]
        cols = [c for c in wanted if c in df.columns]
        if not cols:
            st.info("No usable pending-signups columns returned from API.")
            return
        df = df[cols].copy().fillna("—")
        rename_map = {}
        for c in cols:
            if c == "id":
                rename_map[c] = "ID"
            elif c == "full_name":
                rename_map[c] = "Full name"
            elif c == "email":
                rename_map[c] = "Email"
            elif c == "role":
                rename_map[c] = "Role"
            elif c == "status":
                rename_map[c] = "Status"
            elif c == "created_at":
                rename_map[c] = "Created at"
        df = df.rename(columns=rename_map)
        _render_paginated_table(df, table_type="pending", page_key="pending_page")

    else:
        st.info("Select a card above to view details.")
