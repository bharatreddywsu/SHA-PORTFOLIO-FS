import os
import base64
import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Page config must be the first Streamlit command
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bharat Reddy — Full Stack Portfolio Assistant",
    page_icon="💡",
    layout="centered",
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# Import your response logic
# ─────────────────────────────────────────────────────────────────────────────
from fullstack_qa import get_response  # make sure to rename your main logic file to `fullstack_qa.py`

# ─────────────────────────────────────────────────────────────────────────────
# Avatar display
# ─────────────────────────────────────────────────────────────────────────────
def show_sha_avatar():
    file_path = "shaavatar.png"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style='text-align:center; margin-bottom:15px;'>
                <img src="data:image/png;base64,{encoded}" width="120"
                     style="border-radius:50%; box-shadow:0 0 15px #7F5AF0;">
                <h2 style='color:#E0E0E0; margin-top:10px;'>Bharat's Full Stack Assistant</h2>
                <p style='color:#AAAAAA;'>Ask me about his skills, projects, work, or fun facts!</p>
            </div>
            """,
            unsafe_allow_html=True
        )

show_sha_avatar()

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            background: linear-gradient(135deg, #0A0F2C 0%, #1B0033 100%);
            color: #E0E0E0;
            font-family: 'Poppins', sans-serif;
        }
        .stTextInput > div > div > input {
            background-color: #1B1B2F;
            color: #E0E0E0;
            border: 1px solid #7F5AF0;
            border-radius: 8px;
            padding: 12px;
        }
        .stButton>button, button[kind="primary"] {
            background-color: #7F5AF0 !important;
            color: #FFFFFF !important;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stMarkdown, .stText {
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize session
# ─────────────────────────────────────────────────────────────────────────────
if "miss_count" not in st.session_state:
    st.session_state["miss_count"] = 0

# ─────────────────────────────────────────────────────────────────────────────
# Main Chat UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 🤔 Ask something about Bharat’s full stack journey:")
user_input = st.text_input("Your Question:", key="main_input")

if user_input:
    with st.spinner("Analyzing stack traces..."):
        response = get_response(user_input)
        st.markdown(f"**Assistant:** {response}")

    # Feedback buttons
    st.markdown("#### Was this answer helpful?")
    col1, col2 = st.columns(2)
    if col1.button("👍", key="like_button"):
        with open("questions_log.txt", "a") as f:
            f.write(f"👍 {user_input}\n")
    if col2.button("👎", key="dislike_button"):
        with open("questions_log.txt", "a") as f:
            f.write(f"👎 {user_input}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    "<center><small>🚀 Powered by LangChain + OpenAI · Built by Bharat Reddy</small></center>",
    unsafe_allow_html=True
)
