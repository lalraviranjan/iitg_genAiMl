import streamlit as st
import tempfile
import time
import os
import re
import pdfplumber
import threading
from dotenv import load_dotenv

from typing import List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

# -----------------------------
# FIX OpenMP WARNING
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# ENV
# -----------------------------
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Medical Report Summarizer",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Report Summarizer")
st.markdown("Upload your medical report PDF to get a simple summary.")

# -----------------------------
# GLOBALS (LAZY LOAD)
# -----------------------------
llm = None
retriever = None

def get_llm():
    global llm
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm

def get_retriever():
    global retriever
    if retriever is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        faiss_path = os.path.join(base_dir, "medical_rag_faiss")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"❌ FAISS not found: {faiss_path}")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vectorstore = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    return retriever

# -----------------------------
# PDF LOADER
# -----------------------------
def load_medical_pdf(pdf_path: str):
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:

            text = page.extract_text()
            if text and len(text.strip()) > 50:
                documents.append(Document(page_content=text))

            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = " | ".join(
                        [cell.strip() if cell else "" for cell in row]
                    )
                    if len(row_text.strip()) > 20:
                        documents.append(Document(page_content=row_text))

    return documents

# -----------------------------
# CLEANING
# -----------------------------
def clean_patient_report(text: str) -> str:
    patterns = [
        r"MC-\d+",
        r"Scan QR code to check.*?authenticity",
        r"Page \d+ of \d+",
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)

    return re.sub(r"\n{3,}", "\n\n", text).strip()

# -----------------------------
# STATE
# -----------------------------
class State(TypedDict):
    patient_report: str
    retrieved_docs: List[str]
    mtn_output: str
    condition_output: str
    patient_summary: str

# -----------------------------
# NODES
# -----------------------------
def retrieve(state: State) -> State:
    retriever = get_retriever()
    docs = retriever.invoke(state["patient_report"])
    return {"retrieved_docs": [doc.page_content for doc in docs]}

def mtn(state: State) -> State:
    llm = get_llm()
    res = llm.invoke(f"Simplify:\n{state['patient_report'][:4000]}")
    return {"mtn_output": res.content}

def condition_mapper(state: State) -> State:
    llm = get_llm()
    res = llm.invoke(f"Analyze:\n{state['patient_report'][:4000]}")
    return {"condition_output": res.content}

def patient_summary(state: State) -> State:
    llm = get_llm()
    res = llm.invoke(
        f"""
Summarize for patient:

{state['mtn_output']}
{state['condition_output']}
"""
    )
    return {"patient_summary": res.content}

# -----------------------------
# GRAPH
# -----------------------------
def get_graph():
    workflow = StateGraph(State)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("mtn", mtn)
    workflow.add_node("condition_mapper", condition_mapper)
    workflow.add_node("summary", patient_summary)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "mtn")
    workflow.add_edge("retrieve", "condition_mapper")
    workflow.add_edge("mtn", "summary")
    workflow.add_edge("condition_mapper", "summary")
    workflow.add_edge("summary", END)

    return workflow.compile()

# -----------------------------
# STREAMING
# -----------------------------
def stream_output(text):
    placeholder = st.empty()
    output = ""

    for word in text.split():
        output += word + " "
        placeholder.markdown(output)
        time.sleep(0.02)

# -----------------------------
# UI FLOW
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    if uploaded_file.type != "application/pdf":
        st.error("❌ Please upload a valid PDF.")
    else:
        st.success("✅ File uploaded")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        with st.spinner("⚙️ Processing..."):

            st.write("✅ Reading PDF...")
            docs = load_medical_pdf(temp_path)

            # 🔥 LIMIT DOCUMENTS
            docs = docs[:20]

            st.write("✅ Extracting text...")
            text = "\n\n".join([doc.page_content for doc in docs])

            # 🔥 LIMIT SIZE
            text = text[:8000]

            cleaned = clean_patient_report(text)

            st.write("✅ Building graph...")
            graph = get_graph()

            st.write("✅ Running AI pipeline...")

            result_container = {}

            def run_graph():
                result_container["data"] = graph.invoke({
                    "patient_report": cleaned,
                    "retrieved_docs": [],
                    "mtn_output": "",
                    "condition_output": "",
                    "patient_summary": ""
                })

            thread = threading.Thread(target=run_graph)
            thread.start()
            thread.join(timeout=60)

            if thread.is_alive():
                st.error("❌ Processing timeout. Try smaller PDF.")
                st.stop()

            result = result_container["data"]

        st.subheader("📋 Patient Summary")
        stream_output(result["patient_summary"])

        os.remove(temp_path)