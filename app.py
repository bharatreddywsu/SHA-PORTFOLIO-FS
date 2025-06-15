import os
import base64
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHA — Bharat’s AI Assistant",
    page_icon="👨‍💻",
    layout="centered",
)

load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# Avatar
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
                <h2 style='color:#E0E0E0; margin-top:10px;'>SHA — Bharat’s Full Stack Assistant</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
show_sha_avatar()

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
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
# Build vector store if needed
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists("sha_vector_store"):
    loader = PyPDFLoader("resume/bharat_resume.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)).save_local("sha_vector_store")

# Load memory
store = FAISS.load_local("sha_vector_store", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1),
    chain_type="stuff",
    retriever=store.as_retriever()
)

# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────
def handle_fun(q):
    if "food" in q:
        return "He runs on Java, JSON, and weekend biryani—strictly in that order."
    if "hobbies" in q or "weekend" in q:
        return "Exploring cloud tech, building projects with React & Spring Boot, and catching tech talks."
    return None

def handle_company(q):
    if "current" in q or "working now" in q:
        return "Bharat is a Full Stack Developer at Comcast (since June 2024)."
    if "jpmorgan" in q:
        return "At JPMorgan Chase, he built secure microservices and React-based apps (Feb 2023 – May 2024)."
    if "dentsu" in q:
        return "At Dentsu, he built Angular-based dashboards and Spring Boot APIs for healthcare systems."
    return None

def handle_education(q):
    if "master" in q:
        return "He completed his Master's in Computer Science at Wichita State University (Aug 2022 – May 2024)."
    if "certification" in q:
        return "He's certified in AWS Developer, Oracle Java, and Microsoft Power BI."
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Chat UI
# ─────────────────────────────────────────────────────────────────────────────
if "miss_count" not in st.session_state:
    st.session_state["miss_count"] = 0

st.markdown("### 💬 Ask SHA anything about Bharat:")
user_input = st.text_input("Your Question:", key="main_input")

if user_input:
    q = user_input.lower()

    for fn in [handle_fun, handle_company, handle_education]:
        resp = fn(q)
        if resp:
            st.markdown(f"**SHA:** {resp}")
            break
    else:
        docs = store.as_retriever().get_relevant_documents(user_input)
        if not docs:
            st.session_state["miss_count"] += 1
            msg = [
                "Hmm, that’s not in my memory yet. Try another question?",
                "Still not finding anything—maybe it’s not in the resume.",
                "Alright, here’s my best guess… but you might want to ask Bharat directly 😄"
            ][min(st.session_state["miss_count"], 2)]
            st.markdown(f"**SHA:** {msg}")
        else:
            st.session_state["miss_count"] = 0
            with st.spinner("SHA is thinking..."):
                answer = qa_chain.run(user_input)
            st.markdown(f"**SHA:** {answer}")

    # Feedback
    st.markdown("#### Was this helpful?")
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
st.markdown("<center><small>🤖 Powered by SHA — Bharat’s Full Stack Assistant</small></center>", unsafe_allow_html=True)
