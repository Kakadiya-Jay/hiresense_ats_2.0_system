# src/pages/auth.py
"""
Login and Signup pages. Lightweight UI code only.
"""

from src.ui.context import (
    st,
    safe_rerun,
    set_user_in_session,
    json,
    extract_user_info_from_login,
)
from src.api.ui_integration.auth_api import login as api_login, signup as api_signup
from src.ui import components


def login_page():
    """
    Login form + handler.

    On successful login:
      - stores token into st.session_state["auth_token"] and ["access_token"]
      - immediately calls components.ensure_me_loaded(force=True) to load /auth/me
      - routes to Admin Dashboard if role == 'admin', otherwise to Recruiter Dashboard
      - triggers safe_rerun() so UI (sidebar / top-right) updates immediately
    """
    import requests
    from src.ui import components
    from src.ui.context import st, safe_rerun

    st.title("Hiresense - Login")

    # simple form
    email = st.text_input("Email", value=st.session_state.get("last_email", ""))
    password = st.text_input("Password", type="password")

    # preserve last typed email so user doesn't need to retype after failed login
    if email:
        st.session_state["last_email"] = email

    if st.button("Sign In"):
        if not email or not password:
            st.error("Please provide both email and password.")
            return

        login_url = f"{components.AUTH_API_BASE_URL}/auth/login"
        payload = {"email": email, "password": password}
        try:
            resp = requests.post(login_url, json=payload, timeout=8)
        except requests.RequestException as e:
            st.error(f"Network error when calling login endpoint: {e}")
            return

        # happy path
        if resp.status_code in (200, 201):
            try:
                data = resp.json()
            except Exception:
                st.error("Login succeeded but server returned unexpected payload.")
                return

            # common token keys: access_token, token, auth_token
            token = (
                data.get("access_token") or data.get("token") or data.get("auth_token")
            )
            if not token:
                # sometimes API returns token under different key or nested; show whole payload for debugging
                st.error(
                    "Login response did not contain an access token. Response: "
                    + str(data)
                )
                return

            # store token(s) in session for other helpers/components to use
            st.session_state["auth_token"] = token
            st.session_state["access_token"] = token
            # optionally store raw login response for debugging
            st.session_state["_raw_login_response"] = data

            st.success("Login successful — loading profile...")

            # Immediately fetch current user from /auth/me. We force the re-fetch to guarantee fresh profile.
            # components.ensure_me_loaded will populate st.session_state['current_user'] or clear auth on 401.
            user = components.ensure_me_loaded(force=True)

            if not user:
                # ensure_me_loaded already rendered messages for unauthorized/network errors.
                # If it returned None but we still have token, at least show a friendly message.
                st.warning(
                    "Logged in but couldn't fetch profile. Try refreshing the page."
                )
                # Leave token set — user may still be able to reload manually
                return

            # route based on role
            role = (user.get("role") or "").lower()
            if role == "admin":
                st.session_state["page"] = "Admin Dashboard"
            else:
                # default to recruiter dashboard for non-admin roles
                st.session_state["page"] = "Recruiter Dashboard"

            # remove Login/Signup visibility is handled by sidebar render (it should check current_user)
            # trigger a rerun to update UI immediately
            safe_rerun()
            return

        # failure path: show server-side message when available
        try:
            err = resp.json()
            # best-effort parsing of typical error shapes
            msg = (
                err.get("detail")
                or err.get("message")
                or err.get("error")
                or err.get("msg")
                or str(err)
            )
        except Exception:
            msg = f"Login failed (status={resp.status_code})"

        st.error(f"Sign in failed: {msg}")


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
        try:
            resp_json, status = api_signup(payload)
        except Exception as e:
            st.error(f"Signup request failed: {e}")
            return
        if status in (200, 201, 202):
            st.success(
                "Signup request received. Our team will verify your data within 2 working days."
            )
        else:
            try:
                st.error(resp_json.get("detail", "Error"))
            except Exception:
                st.error("Signup failed")
