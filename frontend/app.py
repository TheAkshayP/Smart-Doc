import streamlit as st
import requests
import base64

# ---------------------------------------
# CONFIG
# ---------------------------------------
BACKEND_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="SmartDoc", layout="wide")

# ---------------------------------------
# CUSTOM CSS
# ---------------------------------------
def load_css():
    css = """
    <style>

    /* Main font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }

    .user-msg {
        background: #262626;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
        color: white;
        width: fit-content;
        max-width: 90%;
    }

    .bot-msg {
        background: #1e1e1e;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
        border-left: 4px solid #4f8bf9;
        color: #e2e2e2;
        max-width: 90%;
    }

    .pdf-frame {
        border-radius: 10px;
        border: 1px solid #333;
        height: 85vh;
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

# ---------------------------------------
# SESSION INIT
# ---------------------------------------
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_base64" not in st.session_state:
    st.session_state.file_base64 = None

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

# ---------------------------------------
# PDF DISPLAY
# ---------------------------------------
def show_pdf(base64_pdf: str):
    pdf_display = f"""
    <iframe class="pdf-frame"
    src="data:application/pdf;base64,{base64_pdf}" width="100%" height="100%"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# ---------------------------------------
# MAIN UI
# ---------------------------------------
if not st.session_state.doc_uploaded:

    st.markdown("<h1 class='title'>SmartDoc</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Your intelligent document analysis assistant.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "docx"])

    if uploaded_file is not None:
        with st.spinner("Processing your document..."):
            file_bytes = uploaded_file.getvalue()

            if uploaded_file.type == "application/pdf":
                st.session_state.file_base64 = base64.b64encode(file_bytes).decode("utf-8")
            else:
                st.session_state.file_base64 = None

            st.session_state.file_name = uploaded_file.name

            files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
            res = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=600)

            if res.status_code == 200:
                st.session_state.doc_uploaded = True
                st.rerun()
            else:
                st.error("Upload failed.")

else:

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        st.markdown(f"<h4>📄 Document: {st.session_state.file_name}</h4>", unsafe_allow_html=True)

        if st.session_state.file_base64:
            show_pdf(st.session_state.file_base64)
        else:
            st.info("No PDF preview available.")

        if st.button("🗑️ Upload New Document"):
            requests.post(f"{BACKEND_URL}/reset")
            st.session_state.doc_uploaded = False
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        st.markdown("<h3>💬 Chat with SmartDoc</h3>", unsafe_allow_html=True)

        # Render chat
        for q, a in st.session_state.chat_history:
            st.markdown(f"<div class='user-msg'>{q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bot-msg'>{a}</div>", unsafe_allow_html=True)

        question = st.chat_input("Ask something...")
        if question:
            st.session_state.chat_history.append((question, "Thinking..."))
            st.rerun()

        # Fetch answer if last response is "Thinking..."
        if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "Thinking...":
            q = st.session_state.chat_history[-1][0]
            res = requests.post(f"{BACKEND_URL}/ask", json={"question": q})
            ans = res.json().get("answer", "Error.")
            st.session_state.chat_history[-1] = (q, ans)
            st.rerun()
