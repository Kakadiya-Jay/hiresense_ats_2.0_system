# src/pages/admin.py
"""
Admin dashboard page. Uses auth_api functions.
"""

from src.ui.context import st, safe_rerun
from src.api.ui_integration.auth_api import (
    get_pending_signups,
    admin_approve,
    admin_reject,
)
from src.ui.context import render_top_right_compact as render_top_right


def admin_dashboard():
    st.title("Admin Dashboard")
    token = st.session_state.get("access_token")
    if not token:
        st.warning("Please login as admin first")
        return
    try:
        resp_json, status = get_pending_signups()
    except Exception as e:
        st.error(f"Failed to call admin endpoint: {e}")
        return

    if status == 200:
        pending = resp_json or []
        st.write(f"Pending signups: {len(pending)}")
        for u in pending:
            st.write(
                f"ID: {u.get('id')} | Name: {u.get('full_name')} | Email: {u.get('email') | 'N/A'} | Role: {u.get('role')} | Recruiter Role: {u.get('recruiter_role') or 'N/A'} | Business: {u.get('business_name') or 'N/A'}"
            )
            cols = st.columns(3)
            if cols[0].button(f"Approve {u.get('id')}", key=f"approve_{u.get('id')}"):
                try:
                    r_json, r_status = admin_approve(u.get("id"))
                    st.write(r_json)
                except Exception as e:
                    st.error(f"Approve failed: {e}")
            if cols[1].button(f"Reject {u.get('id')}", key=f"reject_{u.get('id')}"):
                try:
                    r_json, r_status = admin_reject(
                        u.get("id"), reason="Rejected via admin UI"
                    )
                    st.write(r_json)
                except Exception as e:
                    st.error(f"Reject failed: {e}")
    else:
        st.error("Failed to fetch pending signups. Ensure your token is admin token.")
