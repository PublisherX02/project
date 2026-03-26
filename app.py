import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io
import base64
import requests
import os

# Internal API key for authenticating with the backend gateway
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "OLEA_INTERNAL_GATEWAY_KEY_2026")
INTERNAL_HEADER = {"X-Internal-Key": INTERNAL_API_KEY}

# --- VOICE ENGINE HELPER FUNCTIONS ---
def text_to_audio_autoplay(text, lang='ar'):
    """Converts text to speech and auto-plays it."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md
    except Exception as e:
        return ""

def correct_insurance_stt(raw_text):
    """
    Hackathon God-Mode : Intercepte les erreurs du Speech-to-Text Google
    et les force dans le vocabulaire de l'assurance tunisienne.
    """
    # Dictionnaire de correction : "Ce que Google entend" -> "Ce que le client voulait dire"
    corrections = {
        "kahraba": "karhba",        # Électricité -> Voiture
        "sayara": "karhba",         # Voiture (MSA) -> Voiture (Tounsi)
        "hadith": "accident",       # Histoire/Accident (MSA) -> Accident
        "hades": "accident",
        "aksidon": "accident",
        "taamin": "assurance",      # Assurance (MSA) -> Assurance
        "ta'min": "assurance",
        "zojaj": "bllar",           # Verre (MSA) -> Vitre/Pare-brise (Tounsi)
        "mraya": "rétroviseur",     # Miroir -> Rétroviseur
        "is3af": "dépannage",       # Secours -> Dépannage
        "moteur": "moteur",
        "parchoc": "parchoc",
        "dharba": "accident",       # Coup -> Accident
        "كهرباء": "كرهبة",          # Fixes Arabic Kahraba -> Karhba
        "كهراباء": "كرهبة",
        "حادث": "كسيدة"             # Optional: MSA to Tounsi
    }
    
    text_lower = raw_text.lower()
    
    # Remplacement ultra-rapide
    for wrong, right in corrections.items():
        text_lower = text_lower.replace(wrong, right)
        
    return text_lower


def transcribe_audio(audio_bytes, language_code="ar-TN"):
    """Converts spoken audio into text with noise reduction."""
    recognizer = sr.Recognizer()
    
    # ENHANCEMENT: Applied from Notebook for noisy hackathon environments
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    audio_file = io.BytesIO(audio_bytes)
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            raw_text = recognizer.recognize_google(audio_data, language=language_code)
            
            # --- HACKATHON STT INTERCEPTOR (Dialect Fixes) ---
            final_text = correct_insurance_stt(raw_text)
                
            return final_text
            
    # ENHANCEMENT: Advanced error catching from notebook
    except sr.UnknownValueError:
        return "⚠️ Could not understand audio. Please speak clearly."
    except sr.RequestError as e:
        return f"⚠️ API Error (Check Wi-Fi): {e}"
    except Exception as e:
        return f"⚠️ Audio error: {str(e)}"

# --- UI CONFIGURATION ---
st.set_page_config(page_title="OLEA Service Client", page_icon="olea.png", layout="centered", initial_sidebar_state="expanded")

# Custom CSS for Authentic WhatsApp Web Styling
st.markdown("""
<style>
    .stApp {
        background-color: #efeae2 !important;
        background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png") !important;
        background-repeat: repeat !important;
        background-blend-mode: multiply;
    }
    .block-container { padding-top: 0rem !important; padding-bottom: 5rem !important; }
    #MainMenu {visibility: hidden;}
    header {background: transparent !important;}
    [data-testid="stChatMessage"] { color: #111111 !important; }
    [data-testid="stChatMessage"] * { color: #111111 !important; }
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(odd) {
        background-color: #ffffff !important; border-radius: 0px 8px 8px 8px; margin-bottom: 10px; padding: 10px; box-shadow: 0 1px 1px rgba(0,0,0,0.1);
    }
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(even) {
        background-color: #d9fdd3 !important; border-radius: 8px 0px 8px 8px; margin-bottom: 10px; padding: 10px; box-shadow: 0 1px 1px rgba(0,0,0,0.1); display: flex; flex-direction: row-reverse;
    }
</style>
""", unsafe_allow_html=True)

user_avatar = "👤"
olea_avatar = "olea.png"

# --- SIDEBAR (TOOLS & SETTINGS) ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_language = st.selectbox("Choose your Dialect:", ["Tunisian Arabic (Tounsi)", "Moroccan (Darija)", "Algerian (Dziri)", "English", "French"])
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        try:
            # Tell backend to clear history too
            requests.delete("http://host.docker.internal:8000/api/chat/clear", timeout=5)
        except Exception:
            pass
        st.rerun()

    if st.checkbox("🔍 Show Security Dashboard"):
        st.markdown("### Live Immutable Audit Log")
        try:
            res = requests.get("http://host.docker.internal:8000/api/admin/logs", headers=INTERNAL_HEADER, timeout=5)
            if res.status_code == 200:
                logs = res.json().get("logs", [])
                st.code("".join(logs) if logs else "No logs yet.", language="shell")
            else:
                st.error("Logging API returned an error.")
        except Exception:
            st.error("Waiting for Secure API Gateway...")
        
    st.divider()
    
    st.header("📎 Attachments & Voice")
    # Feature 2: Voice Microphone
    audio_bytes = st.audio_input("🎙️ Record Voice Note:")
    st.caption("💡 *Astuce : Essayez de parler en arabe le plus possible pour une meilleure reconnaissance.*")
    
    # Feature 1: Vision AI
    st.write("📸 **Upload Crash Photo:**")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# --- MAIN CHAT AREA ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Asslema! Bienvenue chez OLEA. Kifech najem n3awnek lyoum?"}]

# Render History
for message in st.session_state.messages:
    avatar_to_use = olea_avatar if message["role"] == "assistant" else user_avatar
    with st.chat_message(message["role"], avatar=avatar_to_use):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Authenticated Sources"):
                for src in message["sources"]:
                    st.caption(f"✓ {src}")

# --- PROCESS SIDEBAR INPUTS ---
prompt = None

# --- PROCESS VISION VIA API (AUTOMATIC & SILENT) ---
if uploaded_file:
    # Check if we have already processed this specific file to prevent infinite looping
    if "processed_image" not in st.session_state or st.session_state.processed_image != uploaded_file.name:
        with st.spinner("Auto-scanning image for fraud and damage severity..."):
            try:
                base64_img = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                payload = {"base64_img": base64_img, "language": selected_language, "filename": uploaded_file.name}
                
                # Use host.docker.internal for Windows Host Networking Fallback
                response = requests.post("http://host.docker.internal:8000/api/vision", json=payload, headers=INTERNAL_HEADER, timeout=60)
                response.raise_for_status()
                assessment = response.json().get("response", "Analysis failed.")
                
                # Render the assessment SILENTLY (No audio HTML generated)
                with st.chat_message("assistant", avatar=olea_avatar):
                    st.markdown(assessment)
                
                st.session_state.messages.append({"role": "assistant", "content": assessment})
                
                # Mark this file as processed so it doesn't run again when you type a message
                st.session_state.processed_image = uploaded_file.name
                
            except Exception as e:
                st.sidebar.error(f"❌ API Error: {str(e)}")

# --- PROCESS INPUTS ---
prompt = None
is_voice_prompt = False  # The flag to control the audio response!

# 1. Check for Voice Input
if audio_bytes:
    with st.spinner("Listening..."):
        stt_lang = "ar-TN" if "Tunisian" in selected_language else "ar-DZ" if "Algerian" in selected_language else "ar-MA" if "Moroccan" in selected_language else "en-US"
        prompt = transcribe_audio(audio_bytes.getvalue(), language_code=stt_lang)
        if "⚠️" in prompt:
            st.error(prompt)
            prompt = None
        else:
            is_voice_prompt = True

# 2. Check for Text Input (This overwrites voice if both happen)
text_input = st.chat_input("Message...", max_chars=500)
if text_input:
    prompt = text_input
    is_voice_prompt = False

# 3. Execute the Chat
if prompt:
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Imani is processing securely..."):
        try:
            payload = {"message": prompt, "language": selected_language, "is_voice": is_voice_prompt}
            
            # Send to FastAPI Backend using Windows Host Networking Fallback
            response = requests.post("http://host.docker.internal:8000/api/chat", json=payload, headers=INTERNAL_HEADER, timeout=60)
            response.raise_for_status()
            
            response_json = response.json()
            bot_response = response_json.get("response", "No response generated.")
            sources = response_json.get("sources", [])
            
            # ONLY generate audio if the user used the microphone
            audio_html = ""
            if is_voice_prompt:
                tts_lang = 'ar' if 'Arabic' in selected_language or 'Dziri' in selected_language or 'Darija' in selected_language else 'en'
                audio_html = text_to_audio_autoplay(bot_response, lang=tts_lang)
            
            with st.chat_message("assistant", avatar=olea_avatar):
                st.markdown(bot_response)
                if sources:
                    with st.expander("📚 Authenticated Sources"):
                        for src in sources:
                            st.caption(f"✓ {src}")
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "sources": sources
            })
            
        except Exception as e:
            error_msg = f"❌ Network Error: Could not reach the AI Backend. Details: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
