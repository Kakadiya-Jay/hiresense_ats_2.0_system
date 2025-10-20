# streamlit_app.py
"""
Main entry point for HireSense MVP.
Updated to integrate automatic /auth/me fetching, improved session handling,
and mapping of URL query params ?token=... and ?page=... into st.session_state
so links from emails (reset-password) open the correct page.
"""

import streamlit as st
from src.ui import components
from src.ui.context import safe_rerun  # use your cross-version rerun wrapper


# Page imports (keep placeholders if pages not present)
try:
    from src.ui.pages.auth import login_page, signup_page, reset_password_page
except Exception:
    # if reset_password_page isn't present, provide a placeholder so routing doesn't crash
    try:
        from src.ui.pages.auth import login_page, signup_page
    except Exception:

        def login_page():
            st.title("Login page placeholder")

        def signup_page():
            st.title("Signup page placeholder")

    def reset_password_page():
        st.title("Reset password page placeholder")
        st.info(
            "Reset page not implemented in src.ui.pages.auth — add reset_password_page() there."
        )


try:
    from src.ui.pages.recruiter import recruiter_dashboard
except Exception:

    def recruiter_dashboard():
        st.title("Recruiter Dashboard placeholder")


try:
    from src.ui.pages.admin import admin_dashboard
except Exception:

    def admin_dashboard():
        st.title("Admin Dashboard placeholder")


try:
    from src.ui.pages.profile import render_profile_card
except Exception:

    def render_profile_card():
        st.title("Profile placeholder")


# App setup
st.set_page_config(layout="wide", page_title="HireSense (MVP)")

# Load current user into session_state if possible (non-blocking)
try:
    components.ensure_me_loaded()
except Exception:
    pass

# ---------- Map URL query params into session state ----------
# This allows links such as:
#   http://localhost:8501/reset-password?token=<URL_ENCODED_TOKEN>
# or
#   http://localhost:8501/?page=Reset%20Password
# to open the correct page in Streamlit (avoids falling back to Login).
try:
    # st.query_params behaves like a MutableMapping, e.g. st.query_params["page"]
    params = dict(st.query_params) if hasattr(st, "query_params") else {}
    # If a page param is present, prefer it (explicit)
    if "page" in params:
        try:
            # New API: if st.query_params used, it returns scalar or list; handle both
            page_value = (
                params["page"][0]
                if isinstance(params["page"], list)
                else params["page"]
            )
            st.session_state["page"] = page_value
        except Exception:
            pass

    # If a token param is present, assume it's a reset flow:
    if "token" in params:
        try:
            token_value = (
                params["token"][0]
                if isinstance(params["token"], list)
                else params["token"]
            )
            st.session_state["reset_token"] = token_value
            st.session_state["page"] = "Reset Password"
            # ✅ Use your safe rerun to force UI update after param load
            safe_rerun()
        except Exception:
            pass
except Exception:
    # st.query_params may not exist on very old Streamlit versions; ignore safely
    pass

# Sidebar Navigation
with st.sidebar:
    st.markdown("## HireSense")
    st.markdown("### Navigation")

    # Use current_user to decide whether to show Log in / Sign up
    current_user = st.session_state.get("current_user")

    if current_user:
        if st.button("Recruiter Dashboard", key="nav_recruiter"):
            st.session_state["page"] = "Recruiter Dashboard"
        if st.button("Admin Dashboard", key="nav_admin"):
            st.session_state["page"] = "Admin Dashboard"
        if st.button("Profile", key="nav_profile"):
            st.session_state["page"] = "Profile"

        st.markdown("---")
        if st.button("Log out", key="nav_logout"):
            components.logout_and_clear()
    else:
        # Not logged in: show login/signup options
        if st.button("Log in", key="nav_login"):
            st.session_state["page"] = "Login"
        if st.button("Sign up", key="nav_signup"):
            st.session_state["page"] = "Signup"

        # still show navigation (optional — keep so user can open pages even when not logged in)
        if st.button("Recruiter Dashboard", key="nav_recruiter"):
            st.session_state["page"] = "Recruiter Dashboard"
        if st.button("Admin Dashboard", key="nav_admin"):
            st.session_state["page"] = "Admin Dashboard"
        if st.button("Profile", key="nav_profile"):
            st.session_state["page"] = "Profile"

        st.markdown("---")
        st.write("You are not signed in.")

# Default page
if "page" not in st.session_state:
    st.session_state["page"] = "Login"

# Detect new token and auto-fetch /auth/me
prev_token = st.session_state.get("_prev_auth_token")
curr_token = st.session_state.get("auth_token")
if curr_token and curr_token != prev_token:
    try:
        components.get_current_user()
    except Exception:
        # swallow errors but keep behavior resilient
        pass
    st.session_state["_prev_auth_token"] = curr_token

# Ensure current_user is loaded when token exists
if st.session_state.get("auth_token") and not st.session_state.get("current_user"):
    try:
        components.ensure_me_loaded()
    except Exception:
        pass

# Layout: main content + top-right block
col_main, col_right = st.columns([3, 1])
with col_right:
    try:
        components.render_top_right_user_block()
    except Exception:
        # fail gracefully if top-right block errors
        st.write("")

with col_main:
    page = st.session_state.get("page")

    if page == "Login":
        login_page()

    elif page == "Signup":
        signup_page()

    elif page == "Recruiter Dashboard":
        components.ensure_me_loaded()
        recruiter_dashboard()

    elif page == "Admin Dashboard":
        components.ensure_me_loaded()
        user = st.session_state.get("current_user")
        if not user:
            st.warning("You must be logged in to view the Admin Dashboard.")
        elif user.get("role") != "admin":
            st.error("Access denied — admins only.")
        else:
            try:
                admin_dashboard()
            except Exception as e:
                st.error("Failed to load Admin Dashboard. See error details below.")
                st.exception(e)

    elif page == "Profile":
        render_profile_card()

    elif page == "Reset Password":
        # Render the reset password page (reads token from session or query param)
        try:
            reset_password_page()
        except Exception as e:
            st.error("Failed to load Reset Password page. See error details below.")
            st.exception(e)

    else:
        st.info("Unknown page. Please use the sidebar.")
