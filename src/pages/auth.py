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


def login_page():
    st.title("HireSense - Login")
    email = st.text_input("Email", key="login_email_page")
    password = st.text_input("Password", type="password", key="login_password_page")
    if st.button("Login"):
        try:
            resp_json, status = api_login(email, password)

            # after resp_json = ... (the parsed response from backend)
            with st.expander("DEBUG: login response (raw)"):
                st.json(resp_json)

            with st.expander("DEBUG: session_state after login"):
                st.json(
                    {
                        k: v
                        for k, v in st.session_state.items()
                        if "user" in k or "recruiter" in k
                    }
                )

        except Exception as e:
            st.error(f"Login request failed: {e}")
            return
        if status == 200:
            token = resp_json.get("access_token") or resp_json.get("token")
            if token:
                st.session_state["access_token"] = token
            # extract name and role if present
            name, role, recruiter_role = extract_user_info_from_login(
                resp_json, fallback_email=email
            )

            # Store in session_state
            st.session_state["user_name"] = name or "User"
            st.session_state["user_role"] = role or "recruiter"
            if recruiter_role:
                st.session_state["recruiter_role"] = recruiter_role

            set_user_in_session(name, role, recruiter_role)
            st.success("Login successful")
            st.session_state["page"] = "Recruiter Dashboard"
            safe_rerun()
        else:
            try:
                st.error(resp_json.get("detail", "Login failed"))
            except Exception:
                st.error("Login failed")


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
