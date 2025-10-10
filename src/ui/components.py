import requests
import time
import pandas as pd
from typing import Optional, Dict, Any, List

from src.ui.context import st, safe_rerun, get_auth_headers

# Config - adjust if needed
AUTH_API_BASE_URL = "http://127.0.0.1:8000"
ME_TTL_SECONDS = 300  # re-fetch /auth/me after this many seconds (if token present)


# ---------------------------
# Low-level API helpers
# ---------------------------
def _auth_headers() -> Dict[str, str]:
    # Prefer centralized helper if available
    try:
        headers = get_auth_headers()
        if headers:
            return headers
    except Exception:
        pass

    # Fallback: read canonical token key
    token = st.session_state.get("access_token") or st.session_state.get("auth_token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Calls GET /auth/me and returns canonical user dict or None on error.
    - On 401 clears auth_token and current_user.
    """
    token = st.session_state.get("auth_token")
    if not token:
        return None

    url = f"{AUTH_API_BASE_URL}/auth/me"
    try:
        resp = requests.get(url, headers=_auth_headers(), timeout=8)
    except requests.RequestException:
        # network problem - keep cached user if present
        return st.session_state.get("current_user")

    if resp.status_code == 200:
        user = resp.json()
        # Normalize booleans -> python bool (your API returns is_active true/false already)
        st.session_state["current_user"] = user
        st.session_state["me_fetched_at"] = int(time.time())
        return user
    elif resp.status_code in (401, 403):
        # token expired/invalid -> clear auth
        for k in ["auth_token", "current_user", "me_fetched_at"]:
            if k in st.session_state:
                del st.session_state[k]
        safe_rerun()
        return None
    else:
        # other status -> keep cached if exist
        return st.session_state.get("current_user")


def ensure_me_loaded(force: bool = False) -> Optional[Dict[str, Any]]:
    """
    Ensure st.session_state['current_user'] exists and is fresh.
    - If token missing returns None.
    - If cached and fresh returns cached user.
    - If stale or missing, fetches from /auth/me.
    """
    token = st.session_state.get("auth_token")
    if not token:
        return None

    fetched_at = st.session_state.get("me_fetched_at")
    if not force and "current_user" in st.session_state and fetched_at:
        if int(time.time()) - int(fetched_at) < ME_TTL_SECONDS:
            return st.session_state["current_user"]

    with st.spinner("Loading profile..."):
        return get_current_user()


# ---------------------------
# Top-right user block rendering
# ---------------------------
def render_top_right_user_block():
    user = st.session_state.get("current_user")
    if not user:
        # Don't render anything when not logged in (aligns with your mockups)
        return

    # existing rendering logic for logged-in user...
    full_name = (
        user.get("full_name") or user.get("username") or user.get("email", "User")
    )
    role = user.get("role", "").title()
    recruiter_role = user.get("recruiter_role")
    with st.container():
        html = "<div style='text-align:right;padding-right:10px'>"
        html += f"<div style='font-size:16px'>Welcome</div>"
        html += f"<div style='font-weight:600'>{full_name}</div>"
        html += f"<div style='opacity:0.8'>{role}</div>"
        if recruiter_role:
            html += f"<div style='opacity:0.7;font-size:13px'>{recruiter_role}</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


# ---------------------------
# Admin: helper calls
# ---------------------------
def fetch_admin_users() -> List[Dict[str, Any]]:
    """
    GET /admin/users
    returns list of user dicts or [] on error.
    """
    url = f"{AUTH_API_BASE_URL}/admin/users"
    try:
        resp = requests.get(url, headers=_auth_headers(), timeout=10)
    except requests.RequestException:
        st.warning("Network error while loading users.")
        return []

    if resp.status_code == 200:
        try:
            return resp.json()
        except Exception:
            return []
    elif resp.status_code in (401, 403):
        st.error("Unauthorized. Please login again.")
        for k in ["auth_token", "current_user"]:
            if k in st.session_state:
                del st.session_state[k]
        return []
    else:
        st.error(f"Failed to fetch users: {resp.status_code}")
        return []


def toggle_user_active(user_id: int, make_active: bool, timeout: int = 8) -> bool:
    """
    Robust attempt to update /admin/users/{id}/status.
    Tries PATCH, then POST, then PUT if server responds 405.
    Returns True on success.
    """
    try:
        url = f"{AUTH_API_BASE_URL}/admin/users/{user_id}/status"
    except NameError:
        st.error("AUTH_API_BASE_URL is not defined in components.py")
        return False

    payload = {"is_active": bool(make_active)}
    headers = _auth_headers()

    methods = ["patch", "post", "put"]
    last_resp = None
    for method in methods:
        try:
            if method == "patch":
                resp = requests.patch(
                    url, json=payload, headers=headers, timeout=timeout
                )
            elif method == "post":
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=timeout
                )
            else:  # put
                resp = requests.put(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            st.error(
                f"Network error while updating user status ({method.upper()}): {e}"
            )
            return False

        last_resp = resp

        if resp.status_code in (200, 204):
            return True

        # If server responds 405, try next method
        if resp.status_code == 405:
            # try next method in loop
            continue

        # Unauthorized: clear session
        if resp.status_code in (401, 403):
            st.error("Unauthorized (session expired). Please log in again.")
            for k in ("access_token", "auth_token", "current_user"):
                if k in st.session_state:
                    del st.session_state[k]
            return False

        # Other errors: try to parse and show message then stop trying
        try:
            data = resp.json()
            msg = data.get("detail") or data.get("message") or str(data)
        except Exception:
            msg = f"Unexpected error (status={resp.status_code})"
        st.error(f"Failed to update user status ({method.upper()}): {msg}")
        return False

    # If we exit loop without success, show last response info
    if last_resp is not None:
        try:
            d = last_resp.json()
            detail = d.get("detail") or d.get("message") or str(d)
        except Exception:
            detail = f"status={last_resp.status_code}"
        st.error(f"Failed to update user status (all methods tried): {detail}")
    else:
        st.error("Failed to update user status: no response from server.")
    return False


def fetch_pending_signups() -> List[Dict[str, Any]]:
    """
    GET /admin/pending-signups
    """
    url = f"{AUTH_API_BASE_URL}/admin/pending-signups"
    try:
        resp = requests.get(url, headers=_auth_headers(), timeout=10)
    except requests.RequestException:
        st.warning("Network error while loading pending signups.")
        return []

    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code in (401, 403):
        st.error("Unauthorized. Please login again.")
        for k in ["auth_token", "current_user"]:
            if k in st.session_state:
                del st.session_state[k]
        return []
    else:
        st.error(f"Failed to fetch pending signups: {resp.status_code}")
        return []


def admin_approve_user(user_id: int, timeout: int = 8) -> bool:
    """
    POST /admin/approve/{user_id}
    Returns True on success (200/204).
    """
    try:
        url = f"{AUTH_API_BASE_URL}/admin/approve/{user_id}"
    except NameError:
        st.error("AUTH_API_BASE_URL is not defined in components.py")
        return False

    headers = _auth_headers()
    try:
        resp = requests.post(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        st.error(f"Network error while approving user: {e}")
        return False

    if resp.status_code in (200, 204):
        return True

    if resp.status_code in (401, 403):
        st.error("Unauthorized. Please login again.")
        for k in ("access_token", "auth_token", "current_user"):
            if k in st.session_state:
                del st.session_state[k]
        return False

    try:
        data = resp.json()
        msg = data.get("detail") or data.get("message") or str(data)
    except Exception:
        msg = f"Error approving user (status={resp.status_code})"
    st.error(msg)
    return False


def admin_reject_user(user_id: int, timeout: int = 8) -> bool:
    """
    POST /admin/reject/{user_id}
    Returns True on success (200/204).
    """
    try:
        url = f"{AUTH_API_BASE_URL}/admin/reject/{user_id}"
    except NameError:
        st.error("AUTH_API_BASE_URL is not defined in components.py")
        return False

    headers = _auth_headers()
    try:
        resp = requests.post(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        st.error(f"Network error while rejecting user: {e}")
        return False

    if resp.status_code in (200, 204):
        return True

    if resp.status_code in (401, 403):
        st.error("Unauthorized. Please login again.")
        for k in ("access_token", "auth_token", "current_user"):
            if k in st.session_state:
                del st.session_state[k]
        return False

    try:
        data = resp.json()
        msg = data.get("detail") or data.get("message") or str(data)
    except Exception:
        msg = f"Error rejecting user (status={resp.status_code})"
    st.error(msg)
    return False


# ---------------------------
# Admin: UI pieces
# ---------------------------
def admin_cards_and_content():
    """
    Renders the three admin cards and the active content area under them.
    Uses st.session_state['admin_active_card'] to store which card is active.
    Values: "users", "resumes", "pending"
    """
    if "admin_active_card" not in st.session_state:
        st.session_state["admin_active_card"] = None

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    with col1:
        if st.button("Total Number of users", key="card_users"):
            st.session_state["admin_active_card"] = "users"
    with col2:
        if st.button("Total Number of resume processed", key="card_resumes"):
            st.session_state["admin_active_card"] = "resumes"
    with col3:
        if st.button("Total Number of pending approvals", key="card_pending"):
            st.session_state["admin_active_card"] = "pending"

    st.write("---")

    active = st.session_state.get("admin_active_card")
    if active == "users":
        render_admin_users_table()
    elif active == "resumes":
        st.info("This functionality is coming in the future.")
    elif active == "pending":
        render_admin_pending_table()
    else:
        # default placeholder when no card clicked
        st.markdown(
            "<div style='background:#e9e9e9;padding:40px;text-align:center;'>"
            "<strong>Select a card above to view details.</strong></div>",
            unsafe_allow_html=True,
        )


def render_admin_users_table():
    """
    Fetch admin users and render table with Deactivate/Activate buttons per row.
    """
    users = fetch_admin_users()
    if not users:
        st.info("No users found.")
        return

    # Prepare DataFrame for display
    df = pd.DataFrame(users)
    display_df = df[
        ["id", "full_name", "email", "role", "is_active", "created_at"]
    ].copy()
    display_df = display_df.rename(
        columns={
            "id": "ID",
            "full_name": "Full name",
            "email": "Email",
            "role": "Role",
            "is_active": "Active",
            "created_at": "Created at",
        }
    )
    st.dataframe(display_df, use_container_width=True)

    st.write("### Actions")
    # action buttons row-by-row
    for u in users:
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(
                f"**{u.get('full_name')}** — {u.get('email')} — role: {u.get('role')}"
            )
        with cols[1]:
            uid = u.get("id")
            active = bool(u.get("is_active", False))
            if active:
                if st.button(f"Deactivate_{uid}", key=f"deact_{uid}"):
                    ok = toggle_user_active(uid, make_active=False)
                    if ok:
                        # refresh table
                        safe_rerun()
            else:
                if st.button(f"Activate_{uid}", key=f"act_{uid}"):
                    ok = toggle_user_active(uid, make_active=True)
                    if ok:
                        safe_rerun()


def render_admin_pending_table():
    """
    Fetch pending signups and render table with Approve/Reject buttons per row.
    """
    pending = fetch_pending_signups()
    if not pending:
        st.info("No pending signups.")
        return

    df = pd.DataFrame(pending)
    display_df = df[["id", "full_name", "email", "role", "status"]].copy()
    display_df = display_df.rename(
        columns={
            "id": "ID",
            "full_name": "Full name",
            "email": "Email",
            "role": "Role",
            "status": "Status",
        }
    )
    st.dataframe(display_df, use_container_width=True)

    st.write("### Actions")
    for p in pending:
        cols = st.columns([3, 1, 1])
        with cols[0]:
            st.markdown(
                f"**{p.get('full_name')}** — {p.get('email')} — role: {p.get('role')}"
            )
        with cols[1]:
            if st.button(f"Approve_{p.get('id')}", key=f"approve_{p.get('id')}"):
                ok = admin_approve_user(p.get("id"))
                if ok:
                    safe_rerun()
        with cols[2]:
            if st.button(f"Reject_{p.get('id')}", key=f"reject_{p.get('id')}"):
                ok = admin_reject_user(p.get("id"))
                if ok:
                    safe_rerun()


# ---------------------------
# Logout helper
# ---------------------------
def logout_and_clear():
    for k in [
        "access_token",
        "auth_token",
        "current_user",
        "me_fetched_at",
        "user_name",
        "user_role",
        "recruiter_role",
        "_prev_auth_token",
        "admin_active_card",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    # ensure both routing keys are set
    st.session_state["page"] = "Login"
    st.session_state["active_page"] = "Login"
    safe_rerun()  # or safe_rerun() if you have that wrapper
