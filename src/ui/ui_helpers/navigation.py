# src/ui/ui_helpers/navigation.py
from typing import List, Tuple
from src.ui.context import st, safe_rerun, render_top_right_compact
import time as _time

def ensure_page_key():
    if "page" not in st.session_state:
        st.session_state["page"] = "Login"

def render_sidebar_list(pages: List[Tuple[str, str]]):
    """
    Render vertical navigation buttons in the sidebar.
      pages: list of tuples (page_key, label)
    Example: [("Login","Login"), ("Recruiter Dashboard","Recruiter Dashboard")]
    """
    ensure_page_key()
    st.sidebar.title("HireSense")
    st.sidebar.write("")  # spacer
    for key, label in pages:
        btn_key = f"nav_{key}"
        if st.sidebar.button(label, key=btn_key):
            st.session_state["page"] = key
            try:
                st.session_state["_last_nav"] = {"page": key, "ts": _time.time()}
            except Exception:
                pass
            safe_rerun()

def render_top_right_user():
    # wrapper that calls compact render from context (keeps single behaviour)
    render_top_right_compact()
