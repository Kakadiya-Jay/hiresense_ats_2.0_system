# src/api/resume_api.py
"""
Resume / scoring API wrappers. Use these functions from pages.
They return parsed JSON (dict/list) or raise exceptions on network errors.
"""

from src.ui.context import requests, RESUME_BASE, TIMEOUT, get_auth_headers, json


def api_process_resume(file_bytes: bytes, filename: str, metadata: dict = None):
    url = f"{RESUME_BASE}/resume/process_resume"
    files = {"file": (filename, file_bytes, "application/pdf")}
    data = metadata or {}
    headers = get_auth_headers()
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=TIMEOUT)
    try:
        return resp.json()
    except Exception:
        resp.raise_for_status()


def api_score_resume(payload: dict):
    url = f"{RESUME_BASE}/resume/score_resume"
    headers = {"Content-Type": "application/json"}
    headers.update(get_auth_headers())
    resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    try:
        return resp.json()
    except Exception:
        resp.raise_for_status()


def api_score_batch_upload(files_payload, data: dict):
    """
    files_payload: list of tuples for requests.files style (("files", (filename, bytes, mime)), ...)
    data: form-data fields
    """
    url = f"{RESUME_BASE}/resume/score_batch/upload"
    headers = get_auth_headers()
    resp = requests.post(
        url, headers=headers, files=files_payload, data=data, timeout=120
    )
    try:
        return resp.json()
    except Exception:
        resp.raise_for_status()


def api_score_batch(payload: dict):
    url = f"{RESUME_BASE}/resume/score_batch"
    headers = {"Content-Type": "application/json"}
    headers.update(get_auth_headers())
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    try:
        return resp.json()
    except Exception:
        resp.raise_for_status()


def api_list_docs():
    url = f"{RESUME_BASE}/resume/list_docs"
    headers = get_auth_headers()
    resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    try:
        return resp.json()
    except Exception:
        resp.raise_for_status()
