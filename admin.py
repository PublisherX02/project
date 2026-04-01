import streamlit as st
import requests
import os

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "olea_admin_2026")
INTERNAL_HEADER = {"x-internal-key": os.getenv("INTERNAL_API_KEY", "OLEA_INTERNAL_GATEWAY_KEY_2026")}
BACKEND_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="OLEA Admin", page_icon="🔒", layout="wide")

if "admin_auth" not in st.session_state:
    st.session_state.admin_auth = False

if not st.session_state.admin_auth:
    st.title("🔒 OLEA Admin Panel")
    pwd = st.text_input("Admin Password", type="password")
    if st.button("Login"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.admin_auth = True
            st.rerun()
        else:
            st.error("Access Denied")
else:
    st.title("🛡️ Live Security Audit Log")
    if st.button("🔄 Refresh"):
        st.rerun()
    try:
        res = requests.get(f"{BACKEND_URL}/api/admin/logs", headers=INTERNAL_HEADER, timeout=5)
        logs = res.json().get("logs", [])
        st.code("".join(logs), language="shell")
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
    if st.button("🚪 Logout"):
        st.session_state.admin_auth = False
        st.rerun()
