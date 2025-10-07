# streamlit_app.py
# Minimal router — imports shared context and helper navigation

from src.ui.context import st, safe_rerun, set_user_in_session, render_top_right_compact
from src.ui.ui_helpers.navigation import render_sidebar_list, render_top_right_user

# Temporary: import page implementations (you will move your functions into src/pages/)
# For now, if you still have those functions in streamlit_app.py, import them or define small wrappers.
try:
    from src.pages.auth import login_page, signup_page
    from src.pages.recruiter import recruiter_dashboard
    from src.pages.admin import admin_dashboard
except Exception:
    # fallback: if pages not yet split, assume functions exist at top-level (older streamlit_app)
    try:
        from streamlit_app import login_page, signup_page, recruiter_dashboard, admin_dashboard  # noqa: F401
    except Exception:
        # minimal placeholders so app can start
        def login_page():
            st.title("Login page placeholder")
        def signup_page():
            st.title("Signup page placeholder")
        def recruiter_dashboard():
            st.title("Recruiter Dashboard (placeholder)")
        def admin_dashboard():
            st.title("Admin Dashboard (placeholder)")

# Define pages (keys and labels)
PAGES = [
    ("Login", "Login"),
    ("Signup", "Signup"),
    ("Recruiter Dashboard", "Recruiter Dashboard"),
    ("Admin Dashboard", "Admin Dashboard"),
]

# Render navigation (list)
render_sidebar_list(PAGES)

# Ensure default key
if "page" not in st.session_state:
    st.session_state["page"] = "Login"

# Route
page = st.session_state.get("page", "Login")
if page == "Signup":
    signup_page()
elif page == "Login":
    login_page()
elif page == "Recruiter Dashboard":
    # render top-right inside page as well (redundant but explicit)
    render_top_right_user()
    recruiter_dashboard()
elif page == "Admin Dashboard":
    render_top_right_user()
    admin_dashboard()
