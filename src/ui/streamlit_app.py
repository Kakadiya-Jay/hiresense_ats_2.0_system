# src/ui/streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("HireSense — Demo UI (skeleton)")

if st.button("Check server health"):
    try:
        r = requests.get(API_URL + "/health", timeout=5)
        if r.ok:
            st.success(r.json())
        else:
            st.error(f"Health check failed: {r.status_code}")
    except Exception as e:
        st.error(f"Connection error: {e}")

st.write("When pipeline is ready, it will be called here.")
