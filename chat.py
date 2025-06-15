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
        return "Haha, that’s classified! Bharat is more in love with Spring Boot than dating apps."
    if "favorite food" in q:
        return "He runs on Java, React, and a weekly dose of biryani."
    if "age" in q:
        return "Age is just metadata—unless it’s part of the API response 😉."
    if any(w in q for w in ["hobbies", "free time", "weekend"]):
        return "Coding personal projects, diving into cloud architecture, or building something cool with React and Spring Boot."
    if "fruit" in q:
        return "Probably a mango—clean APIs on the outside, rich layers underneath."
    if "island" in q:
        return "Set up a self-hosted Kubernetes cluster and code with ocean views."
    if "emoji" in q:
        return "💻—because I live in the terminal."
    if "favorite framework" in q:
        return "Spring Boot on the backend, ReactJS on the frontend—power couple of the stack."
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
        return "I’m currently at Comcast as a Full Stack Developer (June 2024–Present)."
    if "jpmorgan" in q or "jpmc" in q:
        return (
            "At JPMorgan Chase (Feb 2023–May 2024), I built microservices with Java/Spring Boot, "
            "React-based SPAs, and AWS-integrated backend utilities."
        )
    if "dentsu" in q:
        return (
            "At Dentsu (May 2020–May 2022), I modernized healthcare apps using Spring Boot, Angular, and Azure."
        )
    if any(t in q for t in ["wichita state", "master’s", "mscs"]):
        return "Completed my Master’s in Computer Science at Wichita State University (Aug 2022–May 2024)."
    return None

def handle_tech(q: str) -> Optional[str]:
    if "spring boot" in q or "java" in q:
        return "I build scalable backends using Java 17, Spring Boot, JPA, and Microservices architecture."
    if "react" in q or "reactjs" in q:
        return "I build responsive UIs using ReactJS, Redux, and component-driven architecture."
    if "aws" in q:
        return "I deploy microservices to AWS EC2, Lambda, SQS, and RDS—containerized with Docker."
    if "docker" in q or "kubernetes" in q:
        return "I use Docker & Kubernetes to containerize and orchestrate backend services."
    if any(k in q for k in ["ci/cd", "jenkins", "github"]):
        return "I build CI/CD pipelines with Jenkins, GitHub, and Docker for automated deployments."
    return None

def handle_education(q: str) -> Optional[str]:
    if any(w in q for w in ["master", "wichita state"]):
        return "Master’s in Computer Science from Wichita State University (Aug 2022–May 2024)."
    if "bachelor" in q:
        return "Bachelor’s in Engineering (CS) in 2018 with early Python/OpenCV projects."
    if any(w in q for w in ["certification", "certified"]):
        return (
            "Certifications include: AWS Developer Associate, Microsoft Power BI, "
            "Oracle Certified Professional, and Java Certified Programmer."
        )
    return None

def handle_projects(q: str) -> Optional[str]:
    if "loan" in q or "management system" in q:
        return (
            "At Comcast, I developed a scalable Loan Management System using Spring Boot, ReactJS, and AWS Lambda, "
            "reducing processing time by 25%."
        )
    if any(w in q for w in ["face recognition", "raspberry pi"]):
        return "Developed a Pi-based face-recognition system using OpenCV (2018)."
    return None

def handle_volunteer(q: str) -> Optional[str]:
    if any(w in q for w in ["guinness", "wheelchair"]):
        return "Coordinated a Guinness World Record wheelchair event at Vel Tech (May 2019)."
    return None

def handle_behavioral(q: str) -> Optional[str]:
    if any(w in q for w in ["tell me about a time", "example of", "how did you"]):
        return "Sure—want a system optimization story or a leadership example from my time at JPMorgan?"
    return None

# --- Main response function ---
def get_response(user_input: str) -> str:
    q = user_input.lower()
    for fn in [handle_fun, handle_recruiter, handle_company,
               handle_tech, handle_education, handle_projects,
               handle_volunteer, handle_behavioral]:
        resp = fn(q)
        if resp:
            return resp
    docs = store.as_retriever().get_relevant_documents(user_input)
    if docs:
        return qa_chain.run(user_input)
    return "My circuits are tickled—but I don’t have that one yet! Try another question 😊"
