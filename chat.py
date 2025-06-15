import os
from dotenv import load_dotenv
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Paths
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_DIR = os.path.join(BASE_DIR, "sha_vector_store")

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Build or load FAISS vector store
if not os.path.exists(VECTOR_DIR):
    all_docs = []
    for fname in os.listdir(KNOWLEDGE_PATH):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(KNOWLEDGE_PATH, fname))
            all_docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(all_docs)
    FAISS.from_documents(docs, embeddings).save_local(VECTOR_DIR)

# Load the vector store
store = FAISS.load_local(
    VECTOR_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1),
    chain_type="stuff",
    retriever=store.as_retriever()
)

# --- Handler functions ---
def handle_fun(q: str) -> Optional[str]:
    if any(w in q for w in ["girlfriend", "relationship", "single", "wife", "crush"]):
        return "Haha, that’s classified! Bharat is more in love with data pipelines than dating apps."
    if "favorite food" in q:
        return "He runs on JSON, chai, and weekend biryani—strictly in that order."
    if "age" in q:
        return "Age is just metadata—especially if there’s no timestamp 😉."
    if any(w in q for w in ["hobbies", "free time", "weekend"]):
        return "Debugging tricky pipelines, reading AI papers, and sharing memes with fellow engineers."
    if "fruit" in q:
        return "I’d be a pineapple—tough exterior, sweet insights inside."
    if "island" in q:
        return "I’d build a coconut-powered server farm and live off solar CPU cycles."
    if "emoji" in q:
        return "🤖—because I’m building AI side-kicks for people."
    return None


def handle_recruiter(q: str) -> Optional[str]:
    if any(w in q for w in ["sponsorship", "visa", "work authorization"]):
        return (
            "Bharat is on STEM OPT, authorized to work in the U.S., married and awaiting H4. "
            "Future sponsorship can be discussed based on timelines."
        )
    if "notice period" in q:
        return "About a 2-week notice—flexible for the right opportunity."
    if any(w in q for w in ["salary expectation", "current salary", "expected salary"]):
        return "I’m open and flexible—happy to align on compensation based on role and impact."
    if any(w in q for w in ["relocation", "open to relocation"]):
        return "I’m open to remote, hybrid, or relocation—whatever works best for the team."
    return None


def handle_company(q: str) -> Optional[str]:
    if any(t in q for t in ["current company", "working now"]):
        return "I’m currently at **KLA** as a Data Engineer since May 2024."
    if "dentsu" in q:
        return (
            "At Dentsu (May 2020–May 2022), I built pipelines with ADF, Spark, Kafka, and Power BI. "
            "RAG/LLM tech wasn’t in scope then."
        )
    if any(t in q for t in ["wichita state", "master’s", "mscs"]):
        return "Completed my Master’s in CS at Wichita State University (Aug 2022–May 2024)."
    if "fagron" in q:
        return (
            "At Fagron (Dec 2022–Apr 2024), I built HIPAA-compliant ETL on AWS Glue & Redshift, "
            "and introduced a light-based verification system post-production."
        )
    return None


def handle_tech(q: str) -> Optional[str]:
    if any(k in q for k in ["rag", "retrieval augmented generation"]):
        return "I set up FAISS + OpenAI embeddings at KLA for wafer-defect Q&A."
    if any(k in q for k in ["llm", "large language model"]):
        return "I use LLMs in Databricks for code suggestions, debugging, and contextual Q&A."
    if "airflow" in q:
        return "I built DAGs with sensors, retries, SLA alerts, and email notifications."
    if any(k in q for k in ["kafka", "streaming"]):
        return "I’ve written PySpark Structured Streaming consumers with exactly-once semantics."
    return None


def handle_education(q: str) -> Optional[str]:
    if any(w in q for w in ["master", "wichita state"]):
        return "Master’s in CS from Wichita State University (Aug 2022–May 2024)."
    if "bachelor" in q:
        return "Bachelor’s in Engineering (CS) in 2018 with early Python/OpenCV projects."
    if any(w in q for w in ["certification", "certified"]):
        return "Certs: AWS Solutions Architect, Databricks Data Engineer, Snowflake Data Engineer, Python & SQL."
    return None


def handle_projects(q: str) -> Optional[str]:
    if any(w in q for w in ["sawyer", "pybullet"]):
        return "Built a Sawyer Arm sim in PyBullet (Jan–Mar 2023) with IK and reward-based grasp tests."
    if any(w in q for w in ["face recognition", "raspberry pi"]):
        return "Developed a Pi-based face-recognition system using OpenCV (2018)."
    return None


def handle_volunteer(q: str) -> Optional[str]:
    if any(w in q for w in ["guinness", "wheelchair"]):
        return "Coordinated a Guinness World Record wheelchair event at Vel Tech (May 2019)."
    return None


def handle_behavioral(q: str) -> Optional[str]:
    if any(w in q for w in ["tell me about a time", "example of", "how did you"]):
        return "Sure—want a pipeline optimization story or a leadership example?"
    return None


def get_response(user_input: str) -> str:
    q = user_input.lower()
    # Try each handler
    for fn in [handle_fun, handle_recruiter, handle_company,
               handle_tech, handle_education, handle_projects,
               handle_volunteer, handle_behavioral]:
        resp = fn(q)
        if resp:
            return resp
    # Fallback to RAG
    docs = store.as_retriever().get_relevant_documents(user_input)
    if docs:
        return qa_chain.run(user_input)
    # Final funny fallback
    return "My circuits are tickled—but I don’t have that one yet! Try another question 😊"
