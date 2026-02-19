import streamlit as st
from main import chatbot

# Page Configuration
st.set_page_config(
    page_title="Imani - Insurance Guide",
    page_icon="🟢",
    layout="centered"
)

# Custom CSS for WhatsApp-like styling (Dark-Mode Proof)
st.markdown("""
<style>
    /* Force WhatsApp beige background */
    .stApp, .main {
        background-color: #e5ddd5 !important;
    }
    
    /* Make all chat text Dark/Black */
    .stChatMessage p, .stChatMessage div {
        color: #111111 !important;
    }

    /* Style the chat bubbles */
    .stChatMessage {
        padding: 1rem !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
        border: 1px solid #d1d1d1 !important;
    }
    
    /* User Message Bubble (Green) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #dcf8c6 !important;
        border-top-right-radius: 0 !important;
        margin-left: 20%; /* Push user messages to the right */
    }
    
    /* Assistant Message Bubble (White) */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff !important;
        border-top-left-radius: 0 !important;
        margin-right: 20%; /* Push bot messages to the left */
    }
    
    /* Fix Input Box */
    .stChatInputContainer {
        background-color: #ffffff !important;
        border-radius: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Settings
with st.sidebar:
    st.title("🌍 Settings")
    
    # Dialect Selection
    selected_language = st.selectbox(
        "Select Dialect",
        options=[
            "Tunisian Arabic (Tounsi)",
            "Algerian (Dziri)",
            "Moroccan (Darija)",
            "English (Standard)"
        ],
        index=0
    )
    
    # Clear Chat Button
    if st.button("Clear Chat", type="primary"):
        chatbot.clear_history()
        st.session_state.messages = []
        st.rerun()

# Header
st.title("Imani 🤖")
st.markdown(f"**Current Dialect:** {selected_language}")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if prompt := st.chat_input("Ask Imani..."):
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate Response
    with st.spinner("Imani is typing..."):
        try:
            # Call Backend
            response_data = chatbot.chat(prompt, language=selected_language)
            bot_response = response_data.get("response", "An error occurred.")
            
            # Display Bot Message
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            
            # Add to History
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
