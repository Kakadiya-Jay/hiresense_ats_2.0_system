import streamlit as st
import requests
from src.core.config import settings

API_BASE = "http://127.0.0.1:8000"


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
        resp = requests.post(f"{API_BASE}/auth/signup", json=payload)
        if resp.status_code in (200, 202):
            st.success(
                "Signup request received. Our team will verify your data within 2 working days."
            )
        else:
            st.error(resp.json().get("detail", "Error"))


def login_page():
    st.title("HireSense - Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        resp = requests.post(
            f"{API_BASE}/auth/login", json={"email": email, "password": password}
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state["access_token"] = data["access_token"]
            st.success("Login successful")
        else:
            st.error(resp.json().get("detail", "Login failed"))


def recruiter_dashboard():
    st.title("Recruiter Dashboard")
    st.write("Upload resumes (PDF) and job description to run matching (MVP).")
    token = st.session_state.get("access_token")
    if not token:
        st.warning("Please login first")
        return
    files = st.file_uploader("Resumes (PDF)", accept_multiple_files=True, type=["pdf"])
    jd = st.text_area("Paste JD here")
    if st.button("Run matching"):
        st.info(
            "This is a placeholder: implement matching endpoint to receive files + JD."
        )
        # Example of calling an endpoint:
        # headers = {"Authorization": f"Bearer {token}"}
        # files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
        # resp = requests.post(f"{API_BASE}/api/process_batch", headers=headers, files=files_payload, data={"jd": jd})
        # st.write(resp.json())


def admin_dashboard():
    st.title("Admin Dashboard")
    token = st.session_state.get("access_token")
    if not token:
        st.warning("Please login as admin first")
        return
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{API_BASE}/admin/pending-signups", headers=headers)
    if resp.status_code == 200:
        pending = resp.json()
        st.write(f"Pending signups: {len(pending)}")
        for u in pending:
            st.write(f"ID: {u['id']} | Name: {u['full_name']} | Email: {u['email']}")
            cols = st.columns(3)
            if cols[0].button(f"Approve {u['id']}"):
                r = requests.post(
                    f"{API_BASE}/admin/approve/{u['id']}", headers=headers
                )
                st.write(r.json())
            if cols[1].button(f"Reject {u['id']}"):
                r = requests.post(
                    f"{API_BASE}/admin/reject/{u['id']}",
                    headers=headers,
                    json={"reason": "Rejected via admin UI"},
                )
                st.write(r.json())
    else:
        st.error("Failed to fetch pending signups. Ensure your token is admin token.")


# Simple routing
st.sidebar.title("HireSense")
page = st.sidebar.selectbox(
    "Go to", ["Signup", "Login", "Recruiter Dashboard", "Admin Dashboard"]
)
if page == "Signup":
    signup_page()
elif page == "Login":
    login_page()
elif page == "Recruiter Dashboard":
    recruiter_dashboard()
elif page == "Admin Dashboard":
    admin_dashboard()
