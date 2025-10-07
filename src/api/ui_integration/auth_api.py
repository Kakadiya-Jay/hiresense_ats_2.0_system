# src/api/ui_integration/auth_api.py
"""
Auth API wrappers. Uses shared config from src.ui.context
Functions return parsed JSON on success or raise on error.
"""

from src.ui.context import requests, AUTH_BASE, TIMEOUT, get_auth_headers, json


def login(email: str, password: str):
    url = f"{AUTH_BASE}/auth/login"
    resp = requests.post(
        url, json={"email": email, "password": password}, timeout=TIMEOUT
    )
    try:
        return resp.json(), resp.status_code
    except Exception:
        resp.raise_for_status()


def signup(payload: dict):
    url = f"{AUTH_BASE}/auth/signup"
    resp = requests.post(url, json=payload, timeout=TIMEOUT)
    try:
        return resp.json(), resp.status_code
    except Exception:
        resp.raise_for_status()


def get_pending_signups():
    """
    Admin helper: expects authorization header set via get_auth_headers() in calling code.
    This function uses get_auth_headers() only for convenience if a token exists in session.
    """
    headers = get_auth_headers()
    url = f"{AUTH_BASE}/admin/pending-signups"
    resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    try:
        return resp.json(), resp.status_code
    except Exception:
        resp.raise_for_status()


def admin_approve(uid: str):
    headers = get_auth_headers()
    url = f"{AUTH_BASE}/admin/approve/{uid}"
    resp = requests.post(url, headers=headers, timeout=TIMEOUT)
    try:
        return resp.json(), resp.status_code
    except Exception:
        resp.raise_for_status()


def admin_reject(uid: str, reason: str = "Rejected via admin UI"):
    headers = get_auth_headers()
    url = f"{AUTH_BASE}/admin/reject/{uid}"
    resp = requests.post(url, headers=headers, json={"reason": reason}, timeout=TIMEOUT)
    try:
        return resp.json(), resp.status_code
    except Exception:
        resp.raise_for_status()
