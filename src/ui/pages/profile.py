import streamlit as st
from src.ui import components


def render_profile_card():
    components.ensure_me_loaded()
    user = st.session_state.get("current_user")
    if not user:
        st.info("Please log in to view your profile.")
        return

    # Header with avatar & basic info
    colL, colR = st.columns([1, 3])
    with colL:
        avatar_url = user.get("avatar_url") or user.get("verified_doc_path")
        if avatar_url:
            st.image(avatar_url, width=160)
        else:
            initials = "".join((user.get("full_name", "")[:2] or "U")).upper()
            st.markdown(
                f"<div style='width:140px;height:140px;border-radius:8px;background:#2a2a2a;display:flex;align-items:center;justify-content:center;font-size:36px;color:white'>{initials}</div>",
                unsafe_allow_html=True,
            )

    with colR:
        st.markdown(f"## {user.get('full_name', '—')}")
        st.markdown(
            f"**Email:** [{user.get('email','—')}](mailto:{user.get('email','')})"
        )
        st.markdown(f"**Role:** {user.get('role','—').title()}")
        if user.get("recruiter_role"):
            st.markdown(f"**Recruiter role:** {user.get('recruiter_role')}")
        st.write("")

    # Details card
    with st.container():
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Business name**  \n{user.get('business_name') or '—'}")
            st.markdown(f"**Website**  \n{user.get('website_url') or '—'}")
        with col2:
            st.markdown(f"**Phone**  \n{user.get('phone') or '—'}")
            st.markdown(f"**Member since**  \n{user.get('created_at') or '—'}")
            st.markdown(f"**Active**  \n{ 'Yes' if user.get('is_active') else 'No'}")
        st.markdown("</div>", unsafe_allow_html=True)
