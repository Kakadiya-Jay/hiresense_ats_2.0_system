# src/ui/context.py
"""
Central shared context for UI modules.
- Load config (env or streamlit secrets) once.
- Export shared helpers (safe_rerun, get_auth_headers, set_user_in_session, render_top_right_compact).
- Export commonly used libs (streamlit alias 'st', requests, json, time, pandas).
Other modules should import only from this file, e.g.:
    from src.ui.context import st, requests, get_auth_headers, safe_rerun
"""

# Public imports (import once here and reuse)
import streamlit as st
import requests
import json
import time
import os
import pandas as pd
from typing import Optional, Dict, Any, List


# Config helper (env -> st.secrets -> default)
def _get_secret(key: str, default):
    val = os.getenv(key.upper())
    if val:
        return val
    try:
        s = st.secrets.get(key)
        if s:
            return s
    except Exception:
        pass
    return default


# Shared configuration variables (loaded once)
AUTH_BASE: str = _get_secret("auth_api_base", "http://127.0.0.1:8000")
RESUME_BASE: str = _get_secret("resume_api_base", "http://127.0.0.1:8001")
TIMEOUT: int = int(_get_secret("request_timeout", 30))


# Safe rerun helper (cross-version)
def safe_rerun():
    """
    Try to force a rerun in a cross-version-compatible way.
    - Prefer st.experimental_rerun() if available.
    - Otherwise, do nothing (do NOT mutate URL query params).
    Rationale: using query params as a rerun trick causes fragile race conditions,
    and the new st.query_params API is not a drop-in replacement.
    """
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        # If experimental_rerun exists but errors, swallow and return.
        return

    # Fallback: no-op (widget interactions should auto-rerun the script)
    return


# Auth header helper (reads token from session once)
def get_auth_headers() -> Dict[str, str]:
    token = st.session_state.get("access_token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


# Session helpers for user identity
def set_user_in_session(
    name: Optional[str],
    role: Optional[str] = None,
    recruiter_role: Optional[str] = None,
):
    # Store in session_state
    st.session_state["user_name"] = name or "User"
    st.session_state["user_role"] = role or "recruiter"
    st.session_state["recruiter_role"] = recruiter_role


def clear_user_session():
    for k in ["access_token", "user_name", "user_role"]:
        st.session_state.pop(k, None)


# Render compact top-right user widget (call on page top)
def render_top_right_compact():
    """
    Display logged-in user info at top-right:
    - Name
    - Recruiter role (if any)
    - System role (admin/recruiter)
    """
    name = st.session_state.get("user_name")
    role = st.session_state.get("user_role")
    recruiter_role = st.session_state.get("recruiter_role")

    if not name:
        return
    cols = st.columns([1, 1])
    with cols[1]:
        display = f"**Welcome**  \n**{name}**"
        if recruiter_role:
            display += f"  \n_{recruiter_role}_"
        if role and role.lower() != str(recruiter_role).lower():
            display += f"  \n({role})"
        st.markdown(display)


# src/ui/context.py
def extract_user_info_from_login(resp: dict, fallback_email: Optional[str] = None):
    """
    Extract (name, system_role, recruiter_role) from backend login response.
    Handles nested user objects and multiple field name variations.
    Returns tuple: (display_name, system_role, recruiter_role)
    """
    if not isinstance(resp, dict):
        return (fallback_email, None, None)

    candidates = []
    if "user" in resp and isinstance(resp["user"], dict):
        candidates.append(resp["user"])
    if "data" in resp and isinstance(resp["data"], dict):
        if isinstance(resp["data"].get("user"), dict):
            candidates.append(resp["data"]["user"])
        else:
            candidates.append(resp["data"])
    candidates.append(resp)

    name = None
    role = None
    recruiter_role = None

    for cand in candidates:
        if not isinstance(cand, dict):
            continue

        # Possible name fields from DB
        name = (
            cand.get("full_name")
            or cand.get("name")
            or cand.get("recruiter_name")
            or name
        )

        # System role (admin / recruiter / etc.) from DB
        role = cand.get("role")

        # Recruiter role (from DB)
        recruiter_role = cand.get("recruiter_role") or recruiter_role

        # Boolean flags for admin
        if role is None:
            if cand.get("is_admin") or cand.get("admin") is True:
                role = "admin"

        if name and (role or recruiter_role):
            break

    if not name:
        name = fallback_email

    return (name, role, recruiter_role)


# Exported names for convenience (explicit)
__all__ = [
    "st",
    "requests",
    "json",
    "time",
    "pd",
    "AUTH_BASE",
    "RESUME_BASE",
    "TIMEOUT",
    "safe_rerun",
    "get_auth_headers",
    "set_user_in_session",
    "clear_user_session",
    "render_top_right_compact",
    "_get_secret",
]
