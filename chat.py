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
        return "Haha, that’s classified! Bharat is more in love with Laravel and clean code than dating apps."
    if "favorite food" in q:
        return "He runs on Java, Laravel, and a weekly dose of biryani."
    if "age" in q:
        return "Age is just metadata—unless it’s in the schema 😉."
    if any(w in q for w in ["hobbies", "free time", "weekend"]):
        return "Coding personal projects, exploring cloud platforms, or refining UI components in React."
    if "fruit" in q:
        return "Mango—clean APIs on the outside, rich ORM layers inside."
    if "island" in q:
        return "Self-host Laravel on a Pi, deploy to the cloud, and chill by the sea."
    if "emoji" in q:
        return "💻—because that’s where he lives most of the time."
    if "favorite framework" in q:
        return "Laravel and Spring Boot for the backend, ReactJS for the frontend—a powerful combo."
    return None

def handle_recruiter(q: str) -> Optional[str]:
    if any(w in q for w in ["sponsorship", "visa", "work authorization"]):
        return (
            "Bharat is on STEM OPT and authorized to work in the U.S. Sponsorship can be considered for future roles."
        )
    if "notice period" in q:
        return "Typically a 2-week notice, but flexible for the right opportunity."
    if any(w in q for w in ["salary expectation", "current salary", "expected salary"]):
        return "Open to discussion—Bharat values the right role, team, and impact."
    if any(w in q for w in ["relocation", "open to relocation"]):
        return "Open to remote, hybrid, or relocation roles depending on the opportunity."
    return None

def handle_company(q: str) -> Optional[str]:
    if any(t in q for t in ["current company", "working now", "adroit"]):
        return "Bharat is currently working at Agile Adroit LLC as a Web Developer (since September 2024)."
    if any(t in q for t in ["fagron", "wichita state", "university job"]):
        return "He worked at Fagron Sterile Services (Wichita State University) as a Web Developer from Dec 2022 to May 2024."
    if "capgemini" in q:
        return "At Capgemini (May 2020–May 2022), Bharat developed enterprise applications using Java, Struts, and SQL databases."
    return None

def handle_tech(q: str) -> Optional[str]:
    if "spring boot" in q or "java" in q:
        return "He builds scalable backends using Java 11+, Spring Boot, and Struts with SQL/Oracle databases."
    if "react" in q or "reactjs" in q:
        return "Bharat creates responsive UIs with ReactJS and Angular 4, using modern component architecture."
    if "aws" in q:
        return "He deploys applications on AWS, using services like EC2, RDS, and S3 for hosting and scaling."
    if "docker" in q or "kubernetes" in q:
        return "He containerizes applications with Docker and is familiar with deploying to cloud-based infrastructure."
    if any(k in q for k in ["ci/cd", "jenkins", "github"]):
        return "He manages CI/CD using Git, Bitbucket, and Docker, ensuring reliable, automated deployment pipelines."
    return None

def handle_education(q: str) -> Optional[str]:
    if any(w in q for w in ["master", "wichita state"]):
        return "He earned his Master’s in Computer Science from Wichita State University (Aug 2022 – May 2024)."
    if "bachelor" in q:
        return "Bachelor’s in Computer Science with early experience in Python and data projects."
    if any(w in q for w in ["certification", "certified"]):
        return (
            "Certifications include:\n"
            "- AWS Certified Developer – Associate\n"
            "- Microsoft Power BI Data Analyst\n"
            "- Linux Server Management and Security"
        )
    return None

def handle_projects(q: str) -> Optional[str]:
    if "wordpress" in q or "plugin" in q:
        return "Bharat has built and maintained custom WordPress plugins and themes, integrating with APIs and ensuring WCAG 2.1 accessibility."
    if "employee" in q or "scheduling" in q:
        return "He developed a scheduling and reporting app to track employee clock-ins, schedules, and performance metrics."
    return None

def handle_volunteer(q: str) -> Optional[str]:
    return None  # No specific volunteer experience mentioned in the resume

def handle_behavioral(q: str) -> Optional[str]:
    if any(w in q for w in ["tell me about a time", "example of", "how did you"]):
        return (
            "Sure—Bharat once optimized a reporting tool that improved query efficiency by 30% "
            "using MySQL tuning and modular backend refactoring at Agile Adroit."
        )
    return None

# --- Main response function ---
def get_response(user_input: str) -> str:
    q = user_input.lower()
    for fn in [
        handle_fun, handle_recruiter, handle_company,
        handle_tech, handle_education, handle_projects,
        handle_volunteer, handle_behavioral
    ]:
        resp = fn(q)
        if resp:
            return resp
    docs = store.as_retriever().get_relevant_documents(user_input)
    if docs:
        return qa_chain.run(user_input)
    return "My circuits are tickled—but I don’t have that one yet! Try another question 😊"
