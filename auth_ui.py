"""
auth_ui.py — Supabase Auth Login & Sign-Up UI for Insurance AI Chatbot
Imported and called from app.py as a gate before the main chat UI renders.
"""
import streamlit as st
import os
from supabase import create_client, Client
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# --- Supabase client using ANON key (safe for frontend) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
_sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY) if SUPABASE_URL and SUPABASE_ANON_KEY else None


# --- Pydantic Model for Sign-Up Validation ---
class ClientProfile(BaseModel):
    first_name: str = Field(min_length=2, max_length=50)
    last_name: str = Field(min_length=2, max_length=50)
    email: EmailStr
    profession: str = Field(min_length=2, max_length=100)
    income: float = Field(ge=0, description="Monthly income (must be 0 or positive)")
    social_status: Literal["single", "married", "divorced", "widowed"]
    kids: int = Field(default=0, ge=0)
    cars: int = Field(default=1, ge=0)

    @validator("kids")
    def kids_requires_non_single(cls, v, values):
        status = values.get("social_status")
        if v > 0 and status == "single":
            raise ValueError("Single status cannot have kids > 0")
        return v


def _get_supabase() -> Client:
    if not _sb:
        st.error("❌ Supabase is not configured. Check SUPABASE_ANON_KEY in .env")
        st.stop()
    return _sb


def restore_session_from_state():
    """Try to restore a Supabase session from session_state tokens (set after login)."""
    if "sb_access_token" in st.session_state and "sb_refresh_token" in st.session_state:
        try:
            sb = _get_supabase()
            result = sb.auth.set_session(
                st.session_state["sb_access_token"],
                st.session_state["sb_refresh_token"]
            )
            if result.user:
                st.session_state["auth_user"] = result.user
                st.session_state["auth_session"] = result.session
                return True
        except Exception:
            # Token expired — clear and force re-login
            for key in ["sb_access_token", "sb_refresh_token", "auth_user", "auth_session"]:
                st.session_state.pop(key, None)
    return False


def render_auth_gate():
    """
    Main entry point. Call this at the top of app.py.
    If user is authenticated (and email verified), returns without stopping.
    Otherwise renders the login/signup UI and calls st.stop().
    """
    # Already authenticated this run?
    if st.session_state.get("auth_user"):
        return

    # Try restoring from persisted tokens
    if restore_session_from_state():
        return

    # Render the auth page
    _render_auth_page()
    st.stop()


def _render_auth_page():
    imani_b64_path = "Gemini_Generated_Image_olt2tcolt2tcolt2.png"
    import base64
    try:
        imani_b64 = base64.b64encode(open(imani_b64_path, "rb").read()).decode()
        avatar_html = f'<img src="data:image/png;base64,{imani_b64}" style="width:90px;height:90px;border-radius:50%;object-fit:cover;border:3px solid #005C4B;margin-bottom:16px;">'
    except FileNotFoundError:
        avatar_html = '<div style="font-size:48px;margin-bottom:16px;">🛡️</div>'

    st.markdown(f"""
    <style>
    [data-testid="stHeader"], #MainMenu, footer {{ display: none !important; }}
    .stApp {{ background-color: #0B141A !important; }}
    .main .block-container {{
        max-width: 480px !important;
        margin: 8vh auto !important;
        background: #202C33 !important;
        border-radius: 16px !important;
        padding: 40px !important;
        border: 1px solid #2A3942 !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
    }}
    h2 {{ color: #E9EDEF !important; text-align: center; margin-bottom: 4px !important; }}
    p.subtitle {{ color: #8696A0; text-align: center; margin-bottom: 24px; font-size: 14px; }}
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label {{ color: #8696A0 !important; font-size: 13px !important; }}
    .stTextInput input, .stNumberInput input {{ background: #2A3942 !important; color: #E9EDEF !important; border: 1px solid #3D4A52 !important; border-radius: 8px !important; }}
    .stSelectbox > div > div {{ background: #2A3942 !important; color: #E9EDEF !important; border: 1px solid #3D4A52 !important; border-radius: 8px !important; }}
    .stButton > button {{
        background-color: #005C4B !important; color: white !important;
        border-radius: 8px !important; border: none !important;
        font-weight: 600 !important; padding: 10px !important;
        width: 100% !important; margin-top: 8px !important;
    }}
    .stButton > button:hover {{ background-color: #00A884 !important; }}
    .tab-toggle {{ display:flex; gap:8px; margin-bottom:24px; }}
    </style>
    <div style="text-align:center;">{avatar_html}
    <h2>Imani — Insurance AI Assistant</h2>
    <p class="subtitle">Your intelligent insurance companion 🛡️</p></div>
    """, unsafe_allow_html=True)

    # Tab-style toggle
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    col_login, col_signup = st.columns(2)
    with col_login:
        if st.button("🔑 Sign In", use_container_width=True,
                     type="primary" if st.session_state["auth_mode"] == "login" else "secondary"):
            st.session_state["auth_mode"] = "login"
            st.rerun()
    with col_signup:
        if st.button("✨ Create Account", use_container_width=True,
                     type="primary" if st.session_state["auth_mode"] == "signup" else "secondary"):
            st.session_state["auth_mode"] = "signup"
            st.rerun()

    st.markdown("---")

    if st.session_state["auth_mode"] == "login":
        _render_login()
    else:
        _render_signup()


def _render_login():
    st.markdown("<p style='color:#8696A0;text-align:center;font-size:13px;'>Welcome back! Sign in to continue.</p>", unsafe_allow_html=True)
    email = st.text_input("📧 Email address", key="login_email", placeholder="you@example.com")
    password = st.text_input("🔒 Password", type="password", key="login_password", placeholder="Your password")

    if st.button("Sign In →", use_container_width=True):
        if not email or not password:
            st.error("Please fill in both fields.")
            return
        try:
            sb = _get_supabase()
            result = sb.auth.sign_in_with_password({"email": email, "password": password})
            if not result.user:
                st.error("❌ Invalid credentials.")
                return
            if not result.user.email_confirmed_at:
                st.warning("📧 Your email is not verified yet. Please check your inbox.")
                return
            # Fetch profile
            profile = sb.table("profiles").select("*").eq("id", result.user.id).single().execute()
            st.session_state["auth_user"] = result.user
            st.session_state["auth_session"] = result.session
            st.session_state["auth_profile"] = profile.data
            st.session_state["sb_access_token"] = result.session.access_token
            st.session_state["sb_refresh_token"] = result.session.refresh_token
            st.rerun()
        except Exception as e:
            err = str(e)
            if "Invalid login credentials" in err:
                st.error("❌ Incorrect email or password.")
            else:
                st.error(f"❌ Sign-in error: {err}")


def _render_signup():
    st.markdown("<p style='color:#8696A0;text-align:center;font-size:13px;'>Create your account to get started.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        first_name = st.text_input("First Name *", key="su_first", placeholder="Karim")
    with col2:
        last_name = st.text_input("Last Name *", key="su_last", placeholder="Ben Ali")

    email = st.text_input("📧 Email *", key="su_email", placeholder="karim@example.com")

    col3, col4 = st.columns(2)
    with col3:
        password = st.text_input("🔒 Password *", type="password", key="su_pass", placeholder="Min. 6 characters")
    with col4:
        confirm_password = st.text_input("🔒 Confirm Password *", type="password", key="su_pass2", placeholder="Repeat password")

    profession = st.text_input("💼 Profession *", key="su_prof", placeholder="e.g. Engineer, Teacher...")
    income = st.number_input("💰 Monthly Income (TND) *", min_value=0.0, step=100.0, key="su_income")
    social_status = st.selectbox("👤 Marital Status *", ["single", "married", "divorced", "widowed"], key="su_status")
    kids = 0
    if social_status in ["married", "divorced", "widowed"]:
        kids = st.number_input("👶 Number of Children", min_value=0, step=1, key="su_kids")
    cars = st.number_input("🚗 Number of Cars *", min_value=0, step=1, value=1, key="su_cars")

    if st.button("Create Account →", use_container_width=True):
        # Basic checks
        if password != confirm_password:
            st.error("❌ Passwords do not match.")
            return
        if len(password) < 6:
            st.error("❌ Password must be at least 6 characters.")
            return

        # --- Pre-validation: check all required fields are filled ---
        missing = []
        if not first_name or len(first_name.strip()) < 2:
            missing.append("First Name (min 2 characters)")
        if not last_name or len(last_name.strip()) < 2:
            missing.append("Last Name (min 2 characters)")
        if not email or "@" not in email:
            missing.append("Email (must be a valid address)")
        if not profession or len(profession.strip()) < 2:
            missing.append("Profession (min 2 characters)")
        if missing:
            for m in missing:
                st.error(f"❌ Required field missing: **{m}**")
            return

        # Pydantic validation
        try:
            profile_data = ClientProfile(
                first_name=first_name, last_name=last_name, email=email,
                profession=profession, income=income, social_status=social_status,
                kids=int(kids), cars=int(cars)
            )
        except Exception as ve:
            errors = ve.errors() if hasattr(ve, "errors") else [{"msg": str(ve)}]
            for err in errors:
                st.error(f"❌ Validation error: {err['msg']}")
            return

        try:
            sb = _get_supabase()
            # Create Supabase Auth user
            result = sb.auth.sign_up({"email": email, "password": password})
            if not result.user:
                st.error("❌ Account creation failed. Please try again.")
                return

            # Insert profile data (using service role key via backend or anon before RLS triggers)
            # We use the newly created user's session to insert their own profile
            sb_service = create_client(SUPABASE_URL, os.getenv("SUPABASE_KEY"))
            sb_service.table("profiles").insert({
                "id": result.user.id,
                "first_name": profile_data.first_name,
                "last_name": profile_data.last_name,
                "profession": profile_data.profession,
                "income": profile_data.income,
                "social_status": profile_data.social_status,
                "kids": profile_data.kids,
                "cars": profile_data.cars,
            }).execute()

            st.success("✅ Account created! Please check your email to verify your account before signing in.")
            st.info("📧 A verification link has been sent. Click it, then come back and sign in.")
            st.session_state["auth_mode"] = "login"

        except Exception as e:
            err = str(e)
            if "already registered" in err.lower() or "already exists" in err.lower():
                st.error("❌ This email is already registered. Please sign in instead.")
            else:
                st.error(f"❌ Registration error: {err}")
