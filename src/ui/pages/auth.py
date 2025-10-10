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
    st.title("HireSense - Login")
    email = st.text_input("Email", key="login_email_page")
    password = st.text_input("Password", type="password", key="login_password_page")
    if st.button("Sign In", key="do_login"):
        # Call backend login wrapper
        try:
            resp_json, status = api_login(email, password)

        except Exception as e:
            st.error(f"Login request failed: {e}")
            return
        if status == 200 and isinstance(resp_json, dict):
            token = (
                resp_json.get("access_token")
                or resp_json.get("token")
                or resp_json.get("auth_token")
                or resp_json.get("data", {}).get("access_token")
            )
            if not token:
                st.error("Login succeeded but no token returned.")
                return
            st.success("Login successful.")
            # store token in session:

            # canonical key used across app:
            st.session_state["access_token"] = token
            # backward-compatible: keep old key too for modules that still check it
            st.session_state["auth_token"] = token

            # populate legacy session short-fields
            display_name, system_role, recruiter_role = extract_user_info_from_login(
                resp_json, fallback_email=email
            )
            set_user_in_session(display_name, system_role, recruiter_role)

            # immediately fetch full profile into current_user
            try:
                components.get_current_user()  # this helper should rely on get_auth_headers or access_token
            except Exception as e:
                st.warning(f"Failed to fetch profile immediately: {e}")

            # route user
            user = st.session_state.get("current_user") or {"role": system_role}
            role = user.get("role") or system_role or "recruiter"
            if role == "admin":
                st.session_state["page"] = "Admin Dashboard"
                st.session_state["active_page"] = "Admin Dashboard"
            else:
                st.session_state["page"] = "Recruiter Dashboard"
                st.session_state["active_page"] = "Recruiter Dashboard"

            # rerun to update UI immediately
            safe_rerun()
        else:
            # show error returned by API
            try:
                st.error(resp_json.get("detail", "Invalid credentials"))
            except Exception:
                st.error("Login failed. Please check credentials and try again.")


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
