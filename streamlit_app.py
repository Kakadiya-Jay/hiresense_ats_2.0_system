# streamlit_app.py
"""
Main entry point for HireSense MVP.
Updated to integrate automatic /auth/me fetching and improved session handling.
"""

import streamlit as st
from src.ui import components

# Page imports
try:
    from src.ui.pages.auth import login_page, signup_page
    from src.ui.pages.recruiter import recruiter_dashboard
    from src.ui.pages.admin import admin_dashboard
    from src.ui.pages.profile import render_profile_card
except Exception:

    def login_page():
        st.title("Login page placeholder")

    def signup_page():
        st.title("Signup page placeholder")

    def recruiter_dashboard():
        st.title("Recruiter Dashboard placeholder")

    def admin_dashboard():
        st.title("Admin Dashboard placeholder")


# App setup
st.set_page_config(layout="wide", page_title="HireSense (MVP)")

# Load current user into session_state if possible (non-blocking)
try:
    components.ensure_me_loaded()
except Exception:
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
    components.get_current_user()
    st.session_state["_prev_auth_token"] = curr_token

# Ensure current_user is loaded when token exists
if st.session_state.get("auth_token") and not st.session_state.get("current_user"):
    components.ensure_me_loaded()

# Layout: main content + top-right block
col_main, col_right = st.columns([3, 1])
with col_right:
    components.render_top_right_user_block()

with col_main:
    page = st.session_state.get("page")

    if page == "Login":
        login_page()

    elif page == "Signup":
        signup_page()

    elif page == "Recruiter Dashboard":
        components.ensure_me_loaded()
        # components.render_top_right_user_block()
        recruiter_dashboard()

    elif page == "Admin Dashboard":
        components.ensure_me_loaded()
        # components.render_top_right_user_block()
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

    else:
        st.info("Unknown page. Please use the sidebar.")
