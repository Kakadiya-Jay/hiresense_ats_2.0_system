# src/ui/pages/auth.py
"""
Login, Signup and Reset Password pages for HireSense.
Replace existing src/ui/pages/auth.py with this file.
"""

from src.ui.context import st, safe_rerun
from src.ui import components
import requests


def login_page():
    """
    Login form + handler.

    On successful login:
      - stores token into st.session_state["auth_token"] and ["access_token"]
      - immediately calls components.ensure_me_loaded(force=True) to load /auth/me
      - routes to Admin Dashboard if role == 'admin', otherwise to Recruiter Dashboard
      - triggers safe_rerun() so UI (sidebar / top-right) updates immediately
    """
    st.title("Hiresense - Login")

    # simple form for sign-in (not using st.form to keep forgot-button behavior simple)
    email = st.text_input("Email", value=st.session_state.get("last_email", ""))
    password = st.text_input("Password", type="password")

    # Preserve last typed email so the forgot-password form can prefill
    if email:
        st.session_state["last_email"] = email

    # Place the forgot-password toggle/button below the password input, right-aligned
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button(
            "Forgot password?", key="forgot_pwd_toggle", help="Reset password"
        ):
            # Toggle an inline forgot-password form
            st.session_state["show_forgot_form"] = not st.session_state.get(
                "show_forgot_form", False
            )
            # Clear any existing reset_token
            st.session_state.pop("reset_token", None)
            safe_rerun()

    # Inline forgot password form (hidden by default)
    if st.session_state.get("show_forgot_form"):
        with st.form("forgot_email_form"):
            prefill = st.session_state.get("last_email", "")
            forgot_email = st.text_input("Enter your account email", value=prefill)
            send_clicked = st.form_submit_button("Send reset email")
        if send_clicked:
            if not forgot_email:
                st.error("Please enter your email address.")
            else:
                api_url = f"{components.AUTH_API_BASE_URL}/auth/forgot-password"
                try:
                    resp = requests.post(
                        api_url, json={"email": forgot_email}, timeout=8
                    )
                except requests.RequestException as e:
                    st.error(f"Network error while calling forgot-password: {e}")
                else:
                    if resp.status_code in (200, 201):
                        st.success(
                            "If an account exists for that email, a reset link has been sent. Check MailHog or your inbox."
                        )
                        # hide form after sending
                        st.session_state["show_forgot_form"] = False
                        safe_rerun()
                    else:
                        try:
                            detail = (
                                resp.json().get("detail")
                                or resp.json().get("message")
                                or resp.text
                            )
                            st.error(f"Failed to send reset email: {detail}")
                        except Exception:
                            st.error(
                                f"Failed to send reset email (status={resp.status_code})"
                            )

    # Sign in button below controls
    if st.button("Sign In", key="sign_in_button"):
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
                st.error(
                    "Login response did not contain an access token. Response: "
                    + str(data)
                )
                return

            # store token(s) in session for other helpers/components to use
            st.session_state["auth_token"] = token
            st.session_state["access_token"] = token
            st.session_state["_raw_login_response"] = data

            st.success("Login successful — loading profile...")

            # Immediately fetch current user from /auth/me.
            user = components.ensure_me_loaded(force=True)

            if not user:
                st.warning(
                    "Logged in but couldn't fetch profile. Try refreshing the page."
                )
                return

            # route based on role
            role = (user.get("role") or "").lower()
            if role == "admin":
                st.session_state["page"] = "Admin Dashboard"
            else:
                st.session_state["page"] = "Recruiter Dashboard"

            safe_rerun()
            return

        # failure path: show server-side message when available
        try:
            err = resp.json()
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
            resp_json, status = components.api_signup(payload)
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


def reset_password_page():
    """
    Reset Password page — reads token from st.query_params or session_state or user input.
    Uses only st.query_params (no experimental_get_query_params) to avoid StreamlitAPIException.
    """
    st.title("Reset your password")

    # Primary: read from st.query_params (new API)
    try:
        qp = st.query_params  # returns a mapping of lists or scalars
    except Exception:
        # If st.query_params somehow not available, fallback to an empty dict
        qp = {}

    # Normalize token: handle list or scalar
    token = None
    if isinstance(qp, dict):
        token_val = qp.get("token")
        # st.query_params usually returns lists for values; handle both
        if isinstance(token_val, list) and token_val:
            token = token_val[0]
        elif isinstance(token_val, str):
            token = token_val

    # Also allow token previously stored in session (set by streamlit_app)
    if not token:
        token = st.session_state.get("reset_token")

    # If still no token, prompt user (they can paste)
    if not token:
        st.info(
            "If you used the link in your email this page should prefill the token. Otherwise paste token below."
        )
        token = st.text_input("Reset token (paste here)")

    with st.form("reset_form"):
        new_password = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Set new password")

    if submitted:
        if not token:
            st.error(
                "Reset token required. Use the link from your email or paste it here."
            )
            return
        if not new_password or new_password != confirm:
            st.error("Please enter matching passwords.")
            return

        api_url = f"{components.AUTH_API_BASE_URL}/auth/reset-password"
        payload = {"token": token, "new_password": new_password}
        try:
            resp = requests.post(api_url, json=payload, timeout=8)
        except Exception as e:
            st.error(f"Network error: {e}")
            return

        if resp.status_code in (200, 201):
            st.success(
                "Password reset successful. Please sign in with your new password. You can close this window now."
            )
            # cleanup and navigate to login
            st.session_state.pop("reset_token", None)
            st.session_state["page"] = "Login"
            safe_rerun()
        else:
            # show backend detail if available
            try:
                st.error(
                    resp.json().get("detail")
                    or resp.json().get("message")
                    or f"Reset failed ({resp.status_code})"
                )
            except Exception:
                st.error(f"Reset failed (status={resp.status_code})")
