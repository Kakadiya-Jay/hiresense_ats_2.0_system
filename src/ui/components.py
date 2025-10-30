# src/ui/components.py
import requests
import time
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

# import streamlit components from your ui context wrapper
from src.ui.context import st, safe_rerun, get_auth_headers

# Config - adjust if needed
AUTH_API_BASE_URL = "http://127.0.0.1:8000"
ME_TTL_SECONDS = 300  # re-fetch /auth/me after this many seconds (if token present)


# ---------------------------
# Low-level API helpers
# ---------------------------
def _auth_headers() -> Dict[str, str]:
    """
    Resolve authorization headers.
    Prefer centralized get_auth_headers() if available; otherwise fallback to session tokens.
    """
    try:
        headers = get_auth_headers()
        if headers:
            return headers
    except Exception:
        pass

    token = st.session_state.get("access_token") or st.session_state.get("auth_token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def api_signup(payload: dict, timeout: int = 10):
    """
    Send recruiter signup payload to backend.
    Returns (response_json_or_text, status_code)
    """
    url = f"{AUTH_API_BASE_URL}/auth/signup"
    headers = {"Content-Type": "application/json"}
    # attach auth headers only if present (signup normally unauthenticated)
    auth = _auth_headers()
    # _auth_headers may include Content-Type; merge without overwriting
    headers.update({k: v for k, v in auth.items() if k not in headers})

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        # return a simplified error so caller can display it
        return {"error": f"Network error: {e}"}, 0

    # try to return parsed JSON when possible
    try:
        body = resp.json()
    except Exception:
        body = {"text": resp.text}
    return body, resp.status_code


# ---------------------------
# /auth/me helpers
# ---------------------------
def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Calls GET /auth/me and returns canonical user dict or None on error.
    On 401/403 will clear auth info from session_state.
    """
    token = st.session_state.get("auth_token") or st.session_state.get("access_token")
    if not token:
        return None

    url = f"{AUTH_API_BASE_URL}/auth/me"
    try:
        resp = requests.get(url, headers=_auth_headers(), timeout=8)
    except requests.RequestException:
        # network problem - keep cached user if present
        return st.session_state.get("current_user")

    if resp.status_code == 200:
        try:
            user = resp.json()
        except Exception:
            user = None
        if user:
            st.session_state["current_user"] = user
            st.session_state["me_fetched_at"] = int(time.time())
            return user
        return st.session_state.get("current_user")

    if resp.status_code in (401, 403):
        # token expired/invalid -> clear auth and rerun
        for k in ["auth_token", "access_token", "current_user", "me_fetched_at"]:
            if k in st.session_state:
                del st.session_state[k]
        safe_rerun()
        return None

    # other status -> do not overwrite cached user
    return st.session_state.get("current_user")


def ensure_me_loaded(force: bool = False) -> Optional[Dict[str, Any]]:
    """
    Ensure st.session_state['current_user'] exists and is fresh.
    - If no token returns None.
    - If cached and fresh returns cached user.
    - If stale or missing, fetches from /auth/me.
    """
    token = st.session_state.get("auth_token") or st.session_state.get("access_token")
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
    """
    Renders the top-right block showing Welcome / name / role.
    Does nothing when no user is in session_state.
    """
    user = st.session_state.get("current_user")
    if not user:
        # Do not render the block when not logged in.
        return

    full_name = (
        user.get("full_name") or user.get("username") or user.get("email", "User")
    )
    role = (user.get("role") or "").title()
    recruiter_role = user.get("recruiter_role")

    # Render right-aligned block
    with st.container():
        html = "<div style='text-align:right;padding-right:8px;'>"
        html += f"<div style='font-size:14px;opacity:0.9;'>Welcome</div>"
        html += f"<div style='font-weight:600;font-size:16px'>{full_name}</div>"
        if role:
            html += f"<div style='opacity:0.75;font-size:13px'>{role}</div>"
        if recruiter_role:
            html += f"<div style='opacity:0.65;font-size:12px'>{recruiter_role}</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


# ---------------------------
# Admin: low-level API calls
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
            return resp.json() or []
        except Exception:
            return []
    elif resp.status_code in (401, 403):
        st.error("Unauthorized. Please login again.")
        for k in ["auth_token", "access_token", "current_user"]:
            if k in st.session_state:
                del st.session_state[k]
        return []
    else:
        st.error(f"Failed to fetch users: {resp.status_code}")
        return []


def fetch_pending_signups() -> List[Dict[str, Any]]:
    """
    GET /admin/pending-signups
    returns list of pending signup dicts or [] on error.
    """
    url = f"{AUTH_API_BASE_URL}/admin/pending-signups"
    try:
        resp = requests.get(url, headers=_auth_headers(), timeout=10)
    except requests.RequestException:
        st.warning("Network error while loading pending signups.")
        return []

    if resp.status_code == 200:
        try:
            return resp.json() or []
        except Exception:
            return []
    elif resp.status_code in (401, 403):
        st.error("Unauthorized. Please login again.")
        for k in ["auth_token", "access_token", "current_user"]:
            if k in st.session_state:
                del st.session_state[k]
        return []
    else:
        st.error(f"Failed to fetch pending signups: {resp.status_code}")
        return []


# ---------------------------
# Robust action helpers (toggle / approve / reject)
# ---------------------------
def _try_request_with_fallbacks(
    methods: List[str],
    url: str,
    headers: Dict[str, str],
    json_body: Optional[Dict] = None,
    timeout: int = 10,
) -> Tuple[Optional[requests.Response], Optional[str]]:
    """
    Attempts several HTTP methods in order returning first non-405 response.
    Returns (response, method_used) or (None, None) on network error.
    """
    for m in methods:
        try:
            m_lower = m.lower()
            if m_lower == "put":
                resp = requests.put(
                    url, headers=headers, json=json_body, timeout=timeout
                )
            elif m_lower == "patch":
                resp = requests.patch(
                    url, headers=headers, json=json_body, timeout=timeout
                )
            elif m_lower == "post":
                resp = requests.post(
                    url, headers=headers, json=json_body, timeout=timeout
                )
            else:
                resp = requests.request(
                    m_lower, url, headers=headers, json=json_body, timeout=timeout
                )
        except requests.RequestException:
            return None, None

        # If server doesn't reject with 405, return this response to caller
        if resp.status_code != 405:
            return resp, m.upper()
        # else try next method
    return None, None


def toggle_user_active(
    user_id: int, make_active: bool, show_message: bool = False, timeout: int = 10
) -> bool:
    """
    Toggle a user's active status via /admin/users/{id}/status.
    Tries PUT -> PATCH -> POST if necessary.
    Returns True on success.
    NOTE: backend expects {"active": <bool>} so we send that key.
    """
    url = f"{AUTH_API_BASE_URL}/admin/users/{user_id}/status"
    # <-- send the key expected by FastAPI endpoint
    payload = {"active": bool(make_active)}
    headers = _auth_headers()
    methods = ["PUT", "PATCH", "POST"]

    # try multiple methods (existing helper)
    resp, used = _try_request_with_fallbacks(
        methods, url, headers=headers, json_body=payload, timeout=timeout
    )
    if resp is None:
        if show_message:
            st.error("Network error while updating user status.")
        return False

    # debug: optionally print server response to help during dev
    try:
        debug_body = resp.json()
    except Exception:
        debug_body = resp.text

    if resp.status_code in (200, 201, 202, 204):
        if show_message:
            st.success(
                f"User {'activated' if make_active else 'deactivated'} (via {used})."
            )
        return True

    if resp.status_code in (401, 403):
        if show_message:
            st.error("Unauthorized. Please login again.")
        for k in ("auth_token", "access_token", "current_user"):
            if k in st.session_state:
                del st.session_state[k]
        return False

    # show backend error detail when present
    if show_message:
        st.error(
            f"Failed to update user status (HTTP {resp.status_code}): {debug_body}"
        )
    return False


def admin_approve_user(
    user_id: int, show_message: bool = False, timeout: int = 10
) -> bool:
    """
    Approve a pending signup (POST /admin/approve/{user_id}).
    Returns True on success.
    """
    url = f"{AUTH_API_BASE_URL}/admin/approve/{user_id}"
    headers = _auth_headers()
    methods = ["POST", "PUT", "PATCH"]

    resp, used = _try_request_with_fallbacks(
        methods, url, headers=headers, json_body=None, timeout=timeout
    )
    if resp is None:
        if show_message:
            st.error("Network error while approving user.")
        return False

    if resp.status_code in (200, 201, 202, 204):
        if show_message:
            st.success(f"User {user_id} approved (via {used}).")
        return True

    if resp.status_code in (401, 403):
        if show_message:
            st.error("Unauthorized. Please login again.")
        for k in ("auth_token", "access_token", "current_user"):
            if k in st.session_state:
                del st.session_state[k]
        return False

    try:
        err = resp.json()
    except Exception:
        err = resp.text or f"HTTP {resp.status_code}"
    if show_message:
        st.error(f"Failed to approve user: {err}")
    return False


def admin_reject_user(
    user_id: int, show_message: bool = False, timeout: int = 10
) -> bool:
    """
    Reject a pending signup (POST /admin/reject/{user_id}).
    Returns True on success.
    """
    url = f"{AUTH_API_BASE_URL}/admin/reject/{user_id}"
    headers = _auth_headers()
    methods = ["POST", "PUT", "PATCH"]

    resp, used = _try_request_with_fallbacks(
        methods, url, headers=headers, json_body=None, timeout=timeout
    )
    if resp is None:
        if show_message:
            st.error("Network error while rejecting user.")
        return False

    if resp.status_code in (200, 201, 202, 204):
        if show_message:
            st.success(f"User {user_id} rejected (via {used}).")
        return True

    if resp.status_code in (401, 403):
        if show_message:
            st.error("Unauthorized. Please login again.")
        for k in ("auth_token", "access_token", "current_user"):
            if k in st.session_state:
                del st.session_state[k]
        return False

    try:
        err = resp.json()
    except Exception:
        err = resp.text or f"HTTP {resp.status_code}"
    if show_message:
        st.error(f"Failed to reject user: {err}")
    return False


# ---------------------------
# Admin: UI pieces (cards + tables)
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
        st.markdown(
            "<div style='background:#e9e9e9;padding:40px;text-align:center;'>"
            "<strong>Select a card above to view details.</strong></div>",
            unsafe_allow_html=True,
        )


def _paginated_slice(
    df: pd.DataFrame, page_key: str = "page_num", page_size: int = 10
) -> Tuple[pd.DataFrame, int, int]:
    """
    Utility to slice a dataframe according to session-state page number.
    Returns (page_df, current_page, total_pages).
    """
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    current = int(st.session_state.get(page_key, 1))
    total = max(1, (len(df) + page_size - 1) // page_size)
    # clamp
    if current < 1:
        current = 1
        st.session_state[page_key] = 1
    if current > total:
        current = total
        st.session_state[page_key] = total
    start = (current - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].reset_index(drop=True), current, total


def render_admin_users_table():
    """
    Fetch admin users and render table with paging & Activate/Deactivate buttons per row.
    If there are many users, pagination prevents huge lists.
    """
    users = fetch_admin_users()
    if not users:
        st.info("No users found.")
        return

    # Page size control (persisted in session to allow paging to keep same size)
    if "admin_items_per_page" not in st.session_state:
        st.session_state["admin_items_per_page"] = 10
    st.selectbox(
        "Rows per page",
        options=[5, 10, 20, 50],
        index=[5, 10, 20, 50].index(st.session_state["admin_items_per_page"]),
        key="admin_items_per_page",
    )

    page_size = int(st.session_state.get("admin_items_per_page", 10))

    df = pd.DataFrame(users)
    # defensive columns
    cols_wanted = [
        c
        for c in ["id", "full_name", "email", "role", "is_active", "created_at"]
        if c in df.columns
    ]
    if not cols_wanted:
        st.info("No user columns returned from API.")
        return
    display_df = df[cols_wanted].copy()
    rename_map = {}
    if "id" in display_df.columns:
        rename_map["id"] = "ID"
    if "full_name" in display_df.columns:
        rename_map["full_name"] = "Full name"
    if "email" in display_df.columns:
        rename_map["email"] = "Email"
    if "role" in display_df.columns:
        rename_map["role"] = "Role"
    if "is_active" in display_df.columns:
        rename_map["is_active"] = "Active"
    if "created_at" in display_df.columns:
        rename_map["created_at"] = "Created at"
    display_df = display_df.rename(columns=rename_map).fillna("—")

    # paging
    page_df, current_page, total_pages = _paginated_slice(
        display_df, page_key="users_page", page_size=page_size
    )
    st.dataframe(page_df, width="stretch", height=350)

    # paging controls
    cprev, cinfo, cnext = st.columns([1, 2, 1])
    with cprev:
        if st.button("⬅️ Prev", key="users_prev", disabled=(current_page == 1)):
            st.session_state["users_page"] = max(1, current_page - 1)
            safe_rerun()
    with cinfo:
        st.markdown(
            f"<p style='text-align:center;'>Page {current_page} / {total_pages}</p>",
            unsafe_allow_html=True,
        )
    with cnext:
        if st.button(
            "Next ➡️", key="users_next", disabled=(current_page >= total_pages)
        ):
            st.session_state["users_page"] = min(total_pages, current_page + 1)
            safe_rerun()

    st.write("### Actions")
    # Render actions for the users on the current page only
    # Map the page_df back to the original user dicts by ID for complete data
    page_ids = set(page_df["ID"].tolist()) if "ID" in page_df.columns else set()
    # create a mapping of id->user
    user_map = {u.get("id"): u for u in users}

    for uid in page_df["ID"].tolist() if "ID" in page_df.columns else []:
        u = user_map.get(uid) or {}
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(
                f"**{u.get('full_name', '—')}** — {u.get('email', '—')} — role: {u.get('role', '—')}"
            )
        with cols[1]:
            active = bool(u.get("is_active", False))
            if active:
                key = f"deact_user_{uid}"
                if st.button("Deactivate", key=key):
                    ok = toggle_user_active(
                        int(uid), make_active=False, show_message=True
                    )
                    if ok:
                        safe_rerun()
            else:
                key = f"act_user_{uid}"
                if st.button("Activate", key=key):
                    ok = toggle_user_active(
                        int(uid), make_active=True, show_message=True
                    )
                    if ok:
                        safe_rerun()


def render_admin_pending_table():
    """
    Fetch pending signups and render table with Approve/Reject actions and pagination.
    """
    pending = fetch_pending_signups()
    if not pending:
        st.info("No pending signups.")
        return

    # page size
    if "admin_items_per_page" not in st.session_state:
        st.session_state["admin_items_per_page"] = 10
    page_size = int(st.session_state.get("admin_items_per_page", 10))

    df = pd.DataFrame(pending)
    cols_wanted = [
        c
        for c in ["id", "full_name", "email", "role", "status", "created_at"]
        if c in df.columns
    ]
    if not cols_wanted:
        st.info("No pending-signups columns returned from API.")
        return
    display_df = df[cols_wanted].copy()
    rename_map = {}
    if "id" in display_df.columns:
        rename_map["id"] = "ID"
    if "full_name" in display_df.columns:
        rename_map["full_name"] = "Full name"
    if "email" in display_df.columns:
        rename_map["email"] = "Email"
    if "role" in display_df.columns:
        rename_map["role"] = "Role"
    if "status" in display_df.columns:
        rename_map["status"] = "Status"
    if "created_at" in display_df.columns:
        rename_map["created_at"] = "Created at"
    display_df = display_df.rename(columns=rename_map).fillna("—")

    page_df, current_page, total_pages = _paginated_slice(
        display_df, page_key="pending_page", page_size=page_size
    )
    st.dataframe(page_df, width="stretch", height=350)

    # paging controls
    cprev, cinfo, cnext = st.columns([1, 2, 1])
    with cprev:
        if st.button("⬅️ Prev", key="pending_prev", disabled=(current_page == 1)):
            st.session_state["pending_page"] = max(1, current_page - 1)
            safe_rerun()
    with cinfo:
        st.markdown(
            f"<p style='text-align:center;'>Page {current_page} / {total_pages}</p>",
            unsafe_allow_html=True,
        )
    with cnext:
        if st.button(
            "Next ➡️", key="pending_next", disabled=(current_page >= total_pages)
        ):
            st.session_state["pending_page"] = min(total_pages, current_page + 1)
            safe_rerun()

    st.write("### Actions")
    # map id->pending dict
    pending_map = {p.get("id"): p for p in pending}
    for pid in page_df["ID"].tolist() if "ID" in page_df.columns else []:
        p = pending_map.get(pid) or {}
        cols = st.columns([4, 1, 1])
        with cols[0]:
            st.markdown(
                f"**{p.get('full_name', '—')}** — {p.get('email', '—')} — role: {p.get('role', '—')}"
            )
        with cols[1]:
            key_approve = f"approve_{pid}"
            if st.button("Approve", key=key_approve):
                ok = admin_approve_user(pid, show_message=True)
                if ok:
                    safe_rerun()
        with cols[2]:
            key_reject = f"reject_{pid}"
            if st.button("Reject", key=key_reject):
                ok = admin_reject_user(pid, show_message=True)
                if ok:
                    safe_rerun()


# ---------------------------
# Logout helper
# ---------------------------
def logout_and_clear():
    """
    Clear known session keys and navigate back to Login page.
    """
    keys = [
        "access_token",
        "auth_token",
        "current_user",
        "me_fetched_at",
        "user_name",
        "user_role",
        "recruiter_role",
        "_prev_auth_token",
        "admin_active_card",
        "users_page",
        "pending_page",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    # Ensure we land on Login page
    st.session_state["page"] = "Login"
    st.session_state["active_page"] = "Login"
    safe_rerun()


def api_signup(payload: dict, timeout: int = 10):
    """
    Send recruiter signup payload to backend.
    Returns (response_json_or_text, status_code)
    """
    url = f"{AUTH_API_BASE_URL}/auth/signup"
    headers = {"Content-Type": "application/json"}
    # attach auth headers only if present (signup normally unauthenticated)
    auth = _auth_headers()
    # _auth_headers may include Content-Type; merge without overwriting
    headers.update({k: v for k, v in auth.items() if k not in headers})

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        # return a simplified error so caller can display it
        return {"error": f"Network error: {e}"}, 0

    # try to return parsed JSON when possible
    try:
        body = resp.json()
    except Exception:
        body = {"text": resp.text}
    return body, resp.status_code
