import streamlit as st
from main import chatbot
import speech_recognition as sr
from gtts import gTTS
import io
from audio_recorder_streamlit import audio_recorder
import base64

# --- VOICE ENGINE HELPER FUNCTIONS ---
def text_to_audio_autoplay(text, lang='ar'):
    """Converts text to speech and returns an HTML audio player that auto-plays."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        # HTML to auto-play the audio invisibly
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md
    except Exception as e:
        return ""

def transcribe_audio(audio_bytes, language_code="ar-TN"):
    """Converts recorded audio bytes to text."""
    recognizer = sr.Recognizer()
    audio_file = io.BytesIO(audio_bytes)
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            # You can dynamically change this to ar-DZ or ar-MA based on the sidebar!
            text = recognizer.recognize_google(audio_data, language=language_code)
            return text
    except sr.UnknownValueError:
        return "⚠️ Could not understand audio."
    except sr.RequestError:
        return "⚠️ Speech service unavailable."
    except Exception as e:
        return f"⚠️ Audio error: {str(e)}"
# Page Configuration
st.set_page_config(
    page_title="OLEA Service Client",
    page_icon="🟢",
    layout="centered",
    initial_sidebar_state="collapsed" # Starts closed for a true mobile app feel!
)

# Custom CSS for Authentic WhatsApp Web Styling
st.markdown("""
<style>
    /* WhatsApp Background */
    .stApp {
        background-color: #efeae2 !important;
        background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png") !important;
        background-repeat: repeat !important;
        background-blend-mode: multiply;
    }
    
    /* Remove default Streamlit whitespace at the top */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 5rem !important;
    }
    
    /* Hide Streamlit Header */
    header {visibility: hidden;}
    
    /* Text Color */
    [data-testid="stChatMessage"] {
        color: #111111 !important;
    }
    [data-testid="stChatMessage"] * {
        color: #111111 !important;
    }

    /* Bubble Styling */
    [data-testid="stChatMessage"] {
        padding: 8px 12px !important;
        border-radius: 7.5px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 0.5px rgba(11,20,26,.13) !important;
        max-width: 80% !important;
        clear: both;
        display: inline-block;
        width: max-content;
    }
    
    /* User Message (Right) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #dcf8c6 !important;
        border-top-right-radius: 0px !important;
        float: right;
        margin-left: auto !important;
    }
    
    /* OLEA Message (Left) */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff !important;
        border-top-left-radius: 0px !important;
        float: left;
        margin-right: auto !important;
    }
    
    /* Bottom Chat Input */
    .stChatInputContainer {
        background-color: #f0f2f5 !important;
        padding: 10px !important;
        border-radius: 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# Generate Avatars (PDPs)
olea_avatar = "https://ui-avatars.com/api/?name=OLEA&background=008069&color=fff&rounded=true&font-size=0.35&bold=true"
user_avatar = "https://ui-avatars.com/api/?name=You&background=cccccc&color=333&rounded=true"

# Authentic WhatsApp Top Header Bar
st.markdown(f"""
    <div style="background-color: #008069; padding: 12px 20px; display: flex; align-items: center; color: white; margin: -1rem -1rem 1rem -1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.2); position: sticky; top: 0; z-index: 999;">
        <img src="{olea_avatar}" width="42" style="border-radius: 50%; margin-right: 15px; border: 1px solid rgba(255,255,255,0.2);">
        <div style="line-height: 1.2;">
            <div style="font-size: 17px; font-weight: 600; font-family: sans-serif;">OLEA Service Client</div>
            <div style="font-size: 13px; color: #e1e1e1; font-family: sans-serif;">En ligne</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Settings
with st.sidebar:
    st.title("🌍 Paramètres")
    selected_language = st.selectbox(
        "Dialect / Langue",
        options=[
            "Tunisian Arabic (Tounsi)",
            "Algerian (Dziri)",
            "Moroccan (Darija)",
            "English (Standard)"
        ],
        index=0
    )
    if st.button("Clear Chat", type="primary"):
        chatbot.clear_history()
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History with a welcome message
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "Asslema! Bienvenue chez OLEA. Kifech najem n3awnek lyoum?"}]

# Display Chat Messages
for message in st.session_state.messages:
    # Assign the correct PDP based on the role
    avatar_to_use = olea_avatar if message["role"] == "assistant" else user_avatar
    with st.chat_message(message["role"], avatar=avatar_to_use):
        st.markdown(message["content"])

# --- MICROPHONE UI ---
st.write("🎙️ Tap to Speak:")
audio_bytes = audio_recorder(text="", recording_color="#e8b923", neutral_color="#008069", icon_size="2x")

# Handle Voice Input
prompt = None
if audio_bytes:
    with st.spinner("Listening..."):
        # Map selected language to Google STT codes
        stt_lang = "ar-TN" if "Tunisian" in selected_language else "ar-DZ" if "Algerian" in selected_language else "ar-MA" if "Moroccan" in selected_language else "en-US"
        prompt = transcribe_audio(audio_bytes, language_code=stt_lang)
        if "⚠️" in prompt:
            st.error(prompt)
            prompt = None # Cancel if error

# Handle Text Input (Fallback)
text_input = st.chat_input("Message...", max_chars=500)
if text_input:
    prompt = text_input

# --- GENERATE RESPONSE ---
if prompt:
    # 1. Render User Message
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Generate OLEA Response
    with st.spinner("OLEA is typing & thinking..."):
        try:
            response_data = chatbot.chat(prompt, language=selected_language)
            bot_response = response_data.get("response", "An error occurred.")
            
            # 3. Generate Audio for the Response
            # Speak in Arabic if an Arabic dialect is selected, otherwise English
            tts_lang = 'ar' if 'Arabic' in selected_language or 'Dziri' in selected_language or 'Darija' in selected_language else 'en'
            audio_html = text_to_audio_autoplay(bot_response, lang=tts_lang)
            
            # 4. Render OLEA Message + Audio Player
            with st.chat_message("assistant", avatar=olea_avatar):
                st.markdown(bot_response)
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
