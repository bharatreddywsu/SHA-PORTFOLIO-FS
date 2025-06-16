import os
import streamlit as st
import base64
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHA — Bharat’s Full Stack Assistant",
    page_icon="💻",
    layout="centered",
)

load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# Optional avatar
# ─────────────────────────────────────────────────────────────────────────────
def show_avatar():
    img_path = "bharat.png"
    if os.path.exists(img_path):
        encoded = base64.b64encode(open(img_path, "rb").read()).decode()
        st.markdown(f"""
            <div style='text-align:center; margin-bottom:15px;'>
                <img src="data:image/png;base64,{encoded}" width="120"
                     style="border-radius:50%; box-shadow:0 0 15px #7F5AF0;">
                <h2 style='color:#E0E0E0; margin-top:10px;'>SHA — Bharat’s Full Stack Assistant</h2>
            </div>
        """, unsafe_allow_html=True)

show_avatar()

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] {
    background: linear-gradient(135deg, #0A0F2C 0%, #1B0033 100%);
    color: #E0E0E0; font-family: 'Poppins', sans-serif;
  }
  .stTextInput > div > div > input {
    background-color: #1B1B2F; color: #E0E0E0;
    border:1px solid #7F5AF0; border-radius:8px; padding:12px;
  }
  .stButton>button, button[kind="primary"] {
    background-color:#7F5AF0!important; color:#FFFFFF!important;
    border-radius:8px; padding:8px 16px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Build or load vector store
# ─────────────────────────────────────────────────────────────────────────────
resume_path       = "knowledge_base/Bharat_FS.pdf"
vector_store_path = "sha_vector_store"

if not os.path.exists(vector_store_path):
    loader     = PyPDFLoader(resume_path)
    pages      = loader.load()
    splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs       = splitter.split_documents(pages)
    embeds     = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    FAISS.from_documents(docs, embeds).save_local(vector_store_path)

store    = FAISS.load_local(
    vector_store_path,
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    allow_dangerous_deserialization=True
)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1),
    chain_type="stuff",
    retriever=store.as_retriever()
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom handlers
# ─────────────────────────────────────────────────────────────────────────────
def handle_fun(q):
    if "food" in q:
        return "Bharat runs on Java, JSON, and weekend biryani—strictly in that order."
    if "hobby" in q or "weekend" in q:
        return "He codes side projects with Spring Boot & React, reads cloud architecture blogs, and loves tech memes."
    return None

def handle_company(q):
    if any(x in q for x in ["comcast", "current", "working"]):
        return "Bharat is currently at Comcast as a Full Stack Developer (since June 2024)."
    if any(x in q for x in ["jpmorgan", "jpmc"]):
        return "At JPMorgan Chase, he built secure, high-throughput microservices and React apps (2023–2024)."
    if "dentsu" in q:
        return "At Dentsu, he modernized healthcare apps with Spring Boot & Angular on Azure."
    return None

def handle_education(q):
    if "master" in q:
        return "He completed his Master’s in CS from Wichita State University (Aug 2022–May 2024)."
    if "certification" in q:
        return "Certifications include: AWS Developer, Oracle Java Programmer, Power BI Analyst."
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
if "miss_count" not in st.session_state:
    st.session_state["miss_count"] = 0

st.markdown("### 💬 Ask SHA anything about Bharat:")
user_input = st.text_input("Your Question:", key="main_input")

if user_input:
    q = user_input.lower()
    # Try handlers first
    for fn in (handle_fun, handle_company, handle_education):
        resp = fn(q)
        if resp:
            st.markdown(f"**SHA:** {resp}")
            break
    else:
        # Fallback to RAG
        docs = store.as_retriever().get_relevant_documents(user_input)
        if not docs:
            st.session_state["miss_count"] += 1
            msg = [
                "Hmm, that’s not in my memory yet. Try another question?",
                "Still not finding anything—maybe it’s not in the resume.",
                "Okay, I give up! Ask Bharat directly 😅"
            ][min(st.session_state["miss_count"], 2)]
            st.markdown(f"**SHA:** {msg}")
        else:
            st.session_state["miss_count"] = 0
            with st.spinner("SHA is thinking..."):
                answer = qa_chain.run(user_input)
            st.markdown(f"**SHA:** {answer}")

    # Feedback buttons
    st.markdown("#### Was this helpful?")
    col1, col2 = st.columns(2)
    if col1.button("👍"):
        open("questions_log.txt","a").write(f"👍 {user_input}\n")
    if col2.button("👎"):
        open("questions_log.txt","a").write(f"👎 {user_input}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<center><small>🤖 Powered by SHA — Bharat’s Full Stack Assistant</small></center>", unsafe_allow_html=True)
