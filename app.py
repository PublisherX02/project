import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io
import base64
import requests
import os
import uuid
import datetime

# --- CONFIG & SECRETS ---
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "OLEA_INTERNAL_GATEWAY_KEY_2026")
INTERNAL_HEADER = {"x-internal-key": INTERNAL_API_KEY}
# Dynamically extract the base URL
BACKEND_URL = os.getenv("API_URL", "http://host.docker.internal:8000").replace("/api/secure_claim", "")

from dotenv import load_dotenv
from auth_ui import render_auth_gate
load_dotenv()

st.set_page_config(page_title="Imani | Insurance AI", page_icon="Gemini_Generated_Image_olt2tcolt2tcolt2.png", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# AUTH GATE — must pass before anything else renders
# ============================================================
render_auth_gate()
# If we reach here, user is authenticated and email-verified
_auth_user = st.session_state.get("auth_user")
_auth_profile = st.session_state.get("auth_profile", {})
_first_name = _auth_profile.get("first_name", "") if _auth_profile else ""


# --- 1. SESSION MANAGEMENT (UUID PER BROWSER SESSION) ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "sidebar_conversations" not in st.session_state:
    st.session_state.sidebar_conversations = []

# --- 2. MULTILINGUAL ISO MAPPING & PLACEHOLDERS ---
LANG_MAPPING = {
    "Tunisian Arabic (Tounsi)": {"stt": "ar-TN", "tts": "ar", "ph": "اسأل إيماني شنوة تحب..."},
    "Moroccan (Darija)": {"stt": "ar-MA", "tts": "ar", "ph": "سول إيماني..."},
    "Algerian (Dziri)": {"stt": "ar-DZ", "tts": "ar", "ph": "أسأل إيماني..."},
    "English": {"stt": "en-US", "tts": "en", "ph": "Type a message..."},
    "French": {"stt": "fr-FR", "tts": "fr", "ph": "Écrivez un message..."}
}

# --- 2.5 LANGUAGE GATE (CINEMATIC OVERLAY) ---
if "language_confirmed" not in st.session_state:
    st.session_state.language_confirmed = False

if not st.session_state.language_confirmed:
    imani_b64 = base64.b64encode(open('Gemini_Generated_Image_olt2tcolt2tcolt2.png','rb').read()).decode()
    st.markdown(f"""
    <style>
    .lang-overlay {{
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(11,20,26,0.85); backdrop-filter: blur(10px);
        z-index: 9998;
    }}
    .block-container {{
        z-index: 10000 !important; position: relative !important; max-width: 500px !important;
        background: #202C33 !important; padding: 40px !important; border-radius: 16px !important;
        margin-top: 15vh !important; border: 1px solid #2A3942 !important; box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
        text-align: center;
    }}
    /* Surgically lift widgets out of the blur zone */
    [data-testid="stMarkdownContainer"] > div:not(.lang-overlay),
    [data-testid="stSelectbox"],
    .stButton {{
        position: relative !important;
        z-index: 10001 !important;
    }}
    [data-testid="stSidebar"], .whatsapp-header, .custom-footer {{ display: none !important; }}
    </style>
    <div class="lang-overlay"></div>
    <div style="text-align:center; margin-bottom: 20px; position: relative; z-index: 10002;">
        <img src="data:image/png;base64,{imani_b64}" style="width:100px;height:100px;border-radius:50%;object-fit:cover;border:3px solid #005C4B;">
        <h2 style="color:#E9EDEF;margin:15px 0 5px;">Imani UI Virtual Agent</h2>
        <p style="color:#8696A0;margin:0 0 20px; font-size: 14px;">Select your preferred language / إختر لغتك</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Needs to be an expanding list per user rule
    chosen = st.selectbox("🌍 Voice/Text Interface Engine:", ["Tunisian Arabic (Tounsi)", "English", "French"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Confirm Choice / تأكيد", use_container_width=True):
        st.session_state.language_confirmed = True
        st.session_state.selected_lang_name = chosen
        name_part = f" {_first_name}" if _first_name else ""
        welcomes = {
            "Tunisian Arabic (Tounsi)": f"Asslema{name_part}! Ana Imani, msa3dtek fi l'assurance 🛡️. Chnou bech n3awnek lyoum?",
            "English": f"Hello{name_part}! I'm Imani, your virtual insurance AI assistant 🛡️. How can I help you today?",
            "French": f"Bonjour{name_part} ! Je suis Imani, votre assistante virtuelle d'assurance 🛡️. Comment puis-je vous aider aujourd'hui ?"
        }
        st.session_state.messages = [{"role": "assistant", "content": welcomes[chosen]}]
        st.rerun()
    st.stop()


# --- 3. VOICE ENGINE HELPER FUNCTIONS ---
def text_to_audio_autoplay(text, lang='ar'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        return md
    except Exception: return ""

def transcribe_audio(audio_bytes, language_code="ar-TN"):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    audio_file = io.BytesIO(audio_bytes)
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language=language_code)
    except Exception as e: return f"⚠️ Audio error: {str(e)}"

# --- 4. API FETCHING LOGIC ---
def fetch_sidebar_history():
    try:
        res = requests.get(f"{BACKEND_URL}/api/conversations", headers=INTERNAL_HEADER, timeout=3)
        if res.status_code == 200:
            return res.json().get("conversations", [])
    except Exception: pass
    return []

def change_session_and_load_history(session_id):
    st.session_state.session_id = session_id
    try:
        res = requests.get(f"{BACKEND_URL}/api/history/{session_id}", headers=INTERNAL_HEADER, timeout=3)
        if res.status_code == 200:
            st.session_state.messages = []
            hist = res.json().get("history", [])
            for row in hist:
                st.session_state.messages.append({"role": "user", "content": row["user_input"]})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": row["response"],
                    "sources": row.get("sources", [])
                })
        else:
            st.session_state.messages = []
    except Exception:
        st.session_state.messages = []

# Ensure we always attempt to load the sidebar sessions
if not st.session_state.sidebar_conversations:
    st.session_state.sidebar_conversations = fetch_sidebar_history()


# --- 5. UI/UX: MODERN WHATSAPP DESKTOP CSS ---
svg_pattern = """
<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
<style>.i { fill: #E9EDEF; opacity: 0.03; }</style>
<path class="i" d="M20,20 h10 v10 h-10 z M50,50 l5,10 l5,-10 z M150,30 q5,5 10,0 t10,0 M80,120 h15 v15 h-15 z M140,150 c5,0 5,10 10,10 t5,-10"/>
<circle class="i" cx="170" cy="170" r="5"/><circle class="i" cx="30" cy="150" r="3"/>
<rect class="i" x="100" y="20" width="8" height="12" rx="2"/><path class="i" d="M120,80 h10 v2 h-10 z M120,84 h10 v2 h-10 z"/>
</svg>
"""
svg_b64 = base64.b64encode(svg_pattern.encode()).decode()
imani_b64 = base64.b64encode(open('Gemini_Generated_Image_olt2tcolt2tcolt2.png','rb').read()).decode()

st.markdown(f"""
<style>
    /* RESTORE HEADER FOR SIDEBAR TOGGLE */
    [data-testid="stHeader"] {{ 
        background-color: transparent !important; 
        height: 50px !important; 
    }}
    /* Hide Streamlit's default Deploy button and menu to clear the top right corner */
    .stAppDeployButton, [data-testid="stToolbar"] {{
        display: none !important; 
    }}
    .stApp > header {{ background: transparent !important; }}

    /* GLOBAL APP BACKGROUND (Right Web View Area) */
    .stApp {{
        background-color: #0B141A !important;
        background-image: url("data:image/svg+xml;base64,{svg_b64}") !important;
        background-attachment: fixed;
    }}
    
    /* SIDEBAR (Allow native collapsing) */
    [data-testid="stSidebar"] {{
        background-color: #111B21 !important;
        border-right: 1px solid #222D34;
        padding-top: 0 !important;
        position: relative !important; /* Required for bottom-centering the language text */
    }}

    /* DYNAMIC RESPONSIVE WIDTHS FOR HEADER & INPUT BAR */
    /* Default (Sidebar open) */
    .whatsapp-header, [data-testid="stHorizontalBlock"]:has([data-testid="stChatInput"]) {{
        left: 336px !important; 
        width: calc(100vw - 336px) !important;
        transition: all 0.3s ease;
    }}
    /* When sidebar is collapsed */
    section[data-testid="stSidebar"][aria-expanded="false"] + section.main .whatsapp-header,
    section[data-testid="stSidebar"][aria-expanded="false"] + section.main [data-testid="stHorizontalBlock"]:has([data-testid="stChatInput"]) {{
        left: 0 !important;
        width: 100vw !important;
    }}

    .whatsapp-header {{
        position: fixed;
        top: 0;
        height: 60px;
        background-color: #202C33;
        z-index: 999;
        display: flex;
        align-items: center;
        padding: 0 16px;
        border-left: 1px solid #222D34;
        box-shadow: 0 1px 3px rgba(11,20,26,0.4);
    }}
    .contact-avatar {{
        width: 40px; height: 40px; border-radius: 50%; background-color: #005C4B; 
        display: flex; align-items: center; justify-content: center; font-size: 20px; color: white; margin-right: 15px;
    }}
    .contact-info h1 {{ margin: 0; font-size: 16px; color: #E9EDEF; font-weight: 500; font-family: -apple-system, system-ui; }}
    .contact-info p {{ margin: 0; font-size: 13px; color: #8696A0; }}

    /* CHAT CONTAINER PADDING */
    .main .block-container {{
        padding-top: 80px !important; /* Space for the new top header */
        padding-bottom: 90px !important; /* Space for the bottom input */
        padding-left: 5% !important;
        padding-right: 5% !important;
    }}

    /* Remove all previous bubble CSS and replace with this */
    [data-testid="stChatMessageContent"] {{
        background-color: #202C33 !important;
        border-radius: 8px !important;
        color: #E9EDEF !important;
        padding: 10px 14px !important;
        max-width: 75% !important;
        display: inline-block !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }}

    /* Avatars enabled */

    /* WHATSAPP BOTTOM INPUT WRAPPER */
    [data-testid="stHorizontalBlock"]:has([data-testid="stChatInput"]) {{
        background-color: #202C33 !important;
        align-items: center !important;
        padding: 10px 24px !important;
        box-sizing: border-box !important;
        position: fixed !important;
        bottom: 0px !important;
        z-index: 1000 !important;
        height: 70px !important;
        gap: 10px !important;
        border-top: 1px solid #222D34;
    }}

    /* Column Sizing for Attachment | Input | Mic */
    [data-testid="stHorizontalBlock"]:has([data-testid="stChatInput"]) > [data-testid="column"]:nth-child(1) {{ flex: 0 0 45px !important; width: 45px !important; min-width: 45px !important; }}
    [data-testid="stHorizontalBlock"]:has([data-testid="stChatInput"]) > [data-testid="column"]:nth-child(2) {{ flex: 1 1 auto !important; }}
    [data-testid="stHorizontalBlock"]:has([data-testid="stChatInput"]) > [data-testid="column"]:nth-child(3) {{ flex: 0 0 45px !important; width: 45px !important; min-width: 45px !important; }}

    /* Form Input Box Simulation */
    div[data-testid="stChatInput"] {{
        background-color: #2A3942 !important;
        border-radius: 8px !important;
        padding: 0 !important;
        border: none !important;
    }}
    div.stChatInputContainer, div[data-testid="stChatInput"] > div {{
        background-color: transparent !important; border: none !important; padding: 0 !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        background-color: transparent !important; color: #E9EDEF !important; padding: 9px 12px !important; border: none !important;
    }}

    /* Hidden Streamlit Extras for Buttons */
    [data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderDropzone"] > div {{ background: transparent !important; border: none !important; min-height:0!important; margin:0!important; padding:0!important; }}
    [data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] small {{ display: none !important; }}
    [data-testid="stFileUploader"]::before {{ content: "📎"; font-size: 18px; color: #8696A0; opacity: 0.7; cursor: pointer; display: flex; align-items: center; justify-content: center; height: 100%; }}
    [data-testid="stAudioInput"] {{ background: transparent !important; border: none !important; }}
    [data-testid="stAudioInput"] span:not(:has(button)) {{ display: none !important; }}
    [data-testid="stAudioInput"] button {{ background: transparent !important; border: none !important; color: #8696A0 !important; font-size: 22px!important; padding:0!important;}}
    [data-testid="stAudioInput"] > div {{ background: transparent !important; border: none !important; }}
    
    /* Make Mobile Responsive - Override Sidebar Fixed Width on Small Screens */
    @media (max-width: 768px) {{
        .whatsapp-header, [data-testid="stHorizontalBlock"]:has([data-testid="stChatInput"]) {{
            left: 0 !important; 
            width: 100vw !important; /* Ensure mobile also recalculates width correctly */
        }}
    }}
    
    .stButton>button {{
        width: 100%; text-align: left !important; background-color: transparent !important; color: #E9EDEF !important; border: none !important; 
        padding: 12px 15px !important; font-size: 15px; border-bottom: 1px solid #222D34 !important; border-radius: 0 !important;
        display: flex; justify-content: flex-start;
    }}
    .stButton>button:hover {{ background-color: #202C33 !important; }}
    
    /* Clean Selectbox in Sidebar */
    .stSelectbox label {{ color: #00A884 !important; font-weight: 600 !important; }}

    /* TYPING ANIMATION (3 Dots) */
    .typing {{
        width: 38px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
    }}
    .typing span {{
        width: 7px;
        height: 7px;
        background-color: #8696A0;
        border-radius: 50%;
        animation: typing-dot 1s infinite alternate;
    }}
    .typing span:nth-child(2) {{ animation-delay: 0.2s; }}
    .typing span:nth-child(3) {{ animation-delay: 0.4s; }}
    @keyframes typing-dot {{
        from {{ opacity: 0.3; transform: scale(0.8); }}
        to {{ opacity: 1; transform: scale(1.1); }}
    }}

    /* RED LOGOUT BUTTON — Absolute Top Right Fix */
    [data-testid="stButton"] button[kind="primary"] {{
        position: fixed !important;
        top: 14px !important;
        right: 20px !important;
        z-index: 99999 !important;
        background-color: #EA4335 !important;
        color: white !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 5px 15px !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
        height: 32px !important;
        width: auto !important;
    }}
    [data-testid="stButton"] button[kind="primary"]:hover {{ background-color: #C62828 !important; }}
</style>
""", unsafe_allow_html=True)


# --- 6. TOP HEADER (RIGHT PANE FIX) ---
_user_display = f"  {_first_name}" if _first_name else ""
st.markdown(f"""
<div class="whatsapp-header" id="wa-header">
    <div class="contact-avatar" style="overflow:hidden;">
        <img src="data:image/png;base64,{imani_b64}" style="width:40px;height:40px;border-radius:50%;object-fit:cover;" onerror="this.style.display='none';this.parentElement.innerHTML='🛡️';">
    </div>
    <div class="contact-info">
        <h1>Imani Assistant — Welcome, {_first_name}!</h1>
        <p>● online</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- RED LOGOUT BUTTON (top-right, global) ---
# Define the dialog function first
@st.dialog("Sign Out")
def logout_dialog():
    st.write("🚪 Are you sure you want to sign out?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, sign out", use_container_width=True):
            from supabase import create_client
            import os
            try:
                sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
                sb.auth.sign_out()
            except Exception:
                pass
            for key in ["auth_user", "auth_session", "auth_profile", "sb_access_token",
                        "sb_refresh_token", "language_confirmed", "messages",
                        "session_id"]:
                st.session_state.pop(key, None)
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.rerun()

# Trigger the dialog when the primary button is clicked
if st.button("🚪 Logout", key="logout_trigger", type="primary"):
    logout_dialog()


# --- 5. SIDEBAR (SESSION HISTORY) ---
with st.sidebar:
    # Sidebar Header mimicking Web Whatsapp profile area
    st.markdown("""
    <div style="background-color: #202C33; padding: 15px 16px; margin: -6rem -1rem 10px -1rem; display: flex; align-items: center; border-bottom: 1px solid #222D34;">
        <div style="width:40px;height:40px;border-radius:50%;background-color:#E9EDEF;margin-right:15px;display:flex;align-items:center;justify-content:center;font-size:20px;">👤</div>
        <div style="color:#FFF; font-weight:bold; font-size: 16px;">Chats</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.processed_image = None
        st.rerun()
        
    st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
    
    # Load recent sessions list
    if len(st.session_state.sidebar_conversations) > 0:
        for s in st.session_state.sidebar_conversations:
            # Clickable history list
            if st.button(f"📅 {s['title'][:25]}...", key=f"hist_{s['id']}"):
                change_session_and_load_history(s['id'])
                st.rerun()
    else:
        st.markdown("<p style='color:#8696A0; text-align:center; padding: 20px 0;'>No history found. Start a new chat.</p>", unsafe_allow_html=True)

    # Add a spacer so the chat history list doesn't get hidden behind the pinned footer
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

    # Language is locked in from the onboarding screen — display only
    confirmed_lang = st.session_state.get("selected_lang_name", "English")
    st.markdown(f"""
    <div style="position: absolute; bottom: 0px; left: 0px; width: 100%; text-align: center; background-color: #111B21; padding-top: 15px; padding-bottom: 25px; border-top: 1px solid #222D34; z-index: 999;">
        <p style='color:#8696A0; font-size:13px; margin: 0;'>🌍 Language: <br> <strong style='color:#00A884'>{confirmed_lang}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    selected_lang_name = confirmed_lang
    lang_codes = LANG_MAPPING.get(selected_lang_name, LANG_MAPPING["English"])

# --- 7. MAIN CHAT AREA ---
# Messages were already set by the language gate with the correct language
# This fallback only fires if session state was cleared mid-session
if not st.session_state.messages:
    lang = st.session_state.get("selected_lang_name", "English")
    name_part = f" {_first_name}" if _first_name else ""
    welcomes = {
        "Tunisian Arabic (Tounsi)": f"Asslema{name_part}! Ana Imani, msa3dtek fi l'assurance 🛡️. Chnou bech n3awnek lyoum?",
        "English": f"Hello{name_part}! I'm Imani, your virtual insurance AI assistant 🛡️. How can I help you today?",
        "French": f"Bonjour{name_part}! Je suis Imani, votre assistante virtuelle d'assurance 🛡️. Comment puis-je vous aider aujourd'hui?"
    }
    st.session_state.messages = [{"role": "assistant", "content": welcomes.get(lang, welcomes["English"])}]

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="Gemini_Generated_Image_olt2tcolt2tcolt2.png" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for src in message["sources"]: st.caption(f"✓ {src}")


# --- 8. PIXEL-PERFECT WHATSAPP FOOTER ---
st.markdown('<div class="custom-footer">', unsafe_allow_html=True)
footer_row = st.columns([0.5, 9, 0.5])

with footer_row[0]:
    uploaded_file = st.file_uploader("📎", type=["jpg","jpeg","png"], label_visibility="collapsed")

with footer_row[1]:
    prompt = st.chat_input(lang_codes.get("ph", "Type a message..."))

with footer_row[2]:
    audio_bytes = st.audio_input("🎙️", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)


# --- 9. PROCESS LOGIC ---
if uploaded_file and st.session_state.processed_image != uploaded_file.name:
    with st.spinner("Vision AI Scan..."):
        try:
            img_b64 = base64.b64encode(uploaded_file.getvalue()).decode()
            payload = {"base64_img": img_b64, "language": selected_lang_name, "session_id": st.session_state.session_id, "filename": uploaded_file.name}
            res = requests.post(f"{BACKEND_URL}/api/vision", json=payload, headers=INTERNAL_HEADER, timeout=60)
            res.raise_for_status()
            assessment = res.json().get("response", "Vision failed.")
            st.session_state.messages.append({"role": "assistant", "content": assessment})
            st.session_state.processed_image = uploaded_file.name
            
            # Auto-update sidebar history if it was a new chat
            st.session_state.sidebar_conversations = fetch_sidebar_history()
            st.rerun()
        except Exception as e: st.error(str(e))

if audio_bytes:
    with st.spinner("🎙️"):
        raw_audio_prompt = transcribe_audio(audio_bytes.getvalue(), language_code=lang_codes["stt"])
        if "⚠️" not in raw_audio_prompt:
            prompt = raw_audio_prompt # Feed into standard chat logic

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)
    
    with st.empty():
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="background-color: #202C33; padding: 8px 12px; border-radius: 8px; display: inline-block;">
                <div class="typing"><span></span><span></span><span></span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        try:
            payload = {"message": prompt, "language": selected_lang_name, "session_id": st.session_state.session_id, "is_voice": (audio_bytes is not None)}
            res = requests.post(f"{BACKEND_URL}/api/chat", json=payload, headers=INTERNAL_HEADER, timeout=60)
            res.raise_for_status()
            
            data = res.json()
            bot_msg = data.get("response", "No response.")
            
            # Voice Output if needed
            audio_md = ""
            if audio_bytes:
                audio_md = text_to_audio_autoplay(bot_msg, lang=lang_codes["tts"])
            
            with st.chat_message("assistant", avatar="Gemini_Generated_Image_olt2tcolt2tcolt2.png"):
                st.markdown(bot_msg)
                if audio_md: st.markdown(audio_md, unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": bot_msg, "sources": data.get("sources", [])})
            
            # Auto-refresh sidebar list after first API response if title was newly generated
            st.session_state.sidebar_conversations = fetch_sidebar_history()
            
            st.rerun()
        except Exception as e: st.error(f"Network error: {str(e)}")
