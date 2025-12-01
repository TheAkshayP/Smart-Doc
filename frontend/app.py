import streamlit as st
import requests

# --- CONFIGURATION ---
BACKEND_URL = "http://127.0.0.1:8000"
# The config.toml file now controls the base theme, so we set layout here.
st.set_page_config(layout="centered", page_title="SMART-DOC")

# --- CSS STYLING ---
def load_css():
    """
    Loads custom CSS for layout and aesthetics. 
    Base colors are now handled by .streamlit/config.toml
    """
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- Layout Control --- */
    div[data-testid="stAppViewContainer"] > section > div:first-child {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        padding: 0;
    }
    :root[style*="--is-chat-view: 1;"] div[data-testid="stAppViewContainer"] > section > div:first-child {
        height: auto;
        justify-content: flex-start;
    }

    /* --- Title & Tagline --- */
    h1 {
        font-size: 4.5rem;
        font-weight: 700;
        letter-spacing: -2.5px;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.8s ease-out;
    }
    .tagline {
        font-size: 1.1rem;
        font-weight: 400;
        color: #888888;
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeIn 0.9s ease-out;
    }

    /* --- File Uploader Card --- */
    div[data-testid="stFileUploader"] {
        border-radius: 16px;
        padding: 2.5rem;
        border: 1px solid #2C2C2C;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: border-color 0.3s, box-shadow 0.3s;
        animation: fadeIn 1s ease-out;
        text-align: center;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #444444;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stFileUploader"] > label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stFileUploader"] div[data-testid="stHelpText"] {
        color: #888888;
        font-size: 0.85rem;
    }
    div[data-testid="stFileUploader"] > div { padding: 0; border: none; }
    div[data-testid="stFileUploader"] > div > p { display: none; }
    div[data-testid="stFileUploader"] > div > button {
        background: #000000;
        color: #EAEAEA;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-size: 0.9rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        margin-top: 1.5rem;
    }
    div[data-testid="stFileUploader"] > div > button:hover {
        background-color: #1A1A1A;
        border-color: #555;
        color: #FFFFFF;
    }

    /* --- Footer --- */
    .footer { position: fixed; bottom: 20px; right: 20px; color: #666666; font-size: 0.85rem; z-index: 99; }

    /* --- Global Chat View Styling --- */
    .stChatMessage { border: 1px solid #2A2A2A; }
    .stButton > button { background-color: #222; border: 1px solid #444; color: #FAFAFA; }
    .stButton > button:hover { background-color: #333; border: 1px solid #555; color: #FFF; }
    div[data-testid="stChatInput"] { border-top: 1px solid #2C2C2C; }

    /* --- Final, Aggressive Fix for "Upload New" Button --- */
    div.light-button-container .stButton > button {
        background: #ffffff !important; /* Use background shorthand to override gradients */
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 0.5rem;
    }
    div.light-button-container .stButton > button:hover {
        background: #f2f2f2 !important;
        color: #000000 !important;
        border-color: #bbbbbb !important;
    }
    div.light-button-container .stButton > button:focus,
    div.light-button-container .stButton > button:active {
        background: #e6e6e6 !important;
        color: #000000 !important;
        border-color: #aaaaaa !important;
        box-shadow: none !important;
    }

    /* --- Hide Default Streamlit Elements --- */
    header, .stToolbar { display: none !important; }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# --- SESSION STATE ---
if "doc_uploaded" not in st.session_state: st.session_state.doc_uploaded = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "file_name" not in st.session_state: st.session_state.file_name = ""

# --- UI RENDERING ---
load_css()
st.markdown('<div class="footer">Made by Akshay</div>', unsafe_allow_html=True)

if not st.session_state.doc_uploaded:
    # --- UPLOAD VIEW ---
    st.markdown("<h1>SMART-DOC</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Intelligent document analysis, simplified.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["pdf", "txt", "docx"],
        help="Limit 200MB per file • PDF, TXT, DOCX"
    )

    if uploaded_file is not None:
        with st.spinner('Processing document...'):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                res = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=600)
                if res.status_code == 200:
                    st.session_state.doc_uploaded = True
                    st.session_state.file_name = uploaded_file.name
                    st.rerun()
                else: st.error("Upload failed. Please ensure the backend is running.")
            except requests.exceptions.RequestException: st.error(f"Connection failed at {BACKEND_URL}.")

else:
    # --- CHAT VIEW ---
    st.markdown('<style>:root{--is-chat-view: 1;}</style>', unsafe_allow_html=True)
    
    st.info(f"Active Document: **{st.session_state.file_name}**")
    
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.chat_history.append((question, "Thinking..."))
        st.rerun()

    if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "Thinking...":
        user_question = st.session_state.chat_history[-1][0]
        try:
            res = requests.post(f"{BACKEND_URL}/ask", json={"question": user_question}, timeout=300)
            if res.status_code == 200:
                answer = res.json().get("answer", "Sorry, I couldn't find an answer.")
                st.session_state.chat_history[-1] = (user_question, answer)
            else:
                st.session_state.chat_history[-1] = (user_question, "Error: Failed to get an answer.")
        except requests.exceptions.RequestException:
            st.session_state.chat_history[-1] = (user_question, "Error: Connection to backend failed.")
        st.rerun()

    # Wrap the button in a div with a specific class to target it with CSS
    st.markdown('<div class="light-button-container">', unsafe_allow_html=True)
    if st.button("🗑️ Upload New Document"):
        try:
            requests.post(f"{BACKEND_URL}/reset", timeout=30)
            st.session_state.doc_uploaded = False
            st.session_state.chat_history = []
            st.session_state.file_name = ""
            st.rerun()
        except requests.exceptions.RequestException: st.error("Failed to reset. Check backend connection.")
    st.markdown('</div>', unsafe_allow_html=True)