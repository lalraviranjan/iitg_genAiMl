import streamlit as st
import tempfile
import time
import os
import re
import pdfplumber
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
st.markdown("Upload your medical report PDF to get a simple AI-generated summary.")

# -----------------------------
# GLOBALS
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
            raise FileNotFoundError(f"FAISS not found: {faiss_path}")

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
# AGENTS (PROMPTS UNCHANGED)
# -----------------------------
def retrieve(state: State) -> State:
    st.info("🔍 Retrieving relevant medical context...")

    try:
        retriever = get_retriever()
        docs = retriever.invoke(state["patient_report"])

        return {
            "retrieved_docs": [doc.page_content for doc in docs]
        }

    except Exception as e:
        st.error(f"Retriever error: {e}")
        return {"retrieved_docs": []}


def mtn(state: State) -> State:
    st.info("🧠 Simplifying medical terminology...")

    try:
        llm = get_llm()

        prompt = f"""
You are a Medical Report Summarizer and Analyzer.

Task:
- Simplify complex medical terms found in the patient’s lab report
  into clear, easy-to-understand language.
- This is for a non-medical patient.

Guidelines:
- Use the retrieved medical references as a SUPPORTING reference.
- Use your own trained medical knowledge extensively to explain terms clearly.
- Do NOT introduce information that is not medically relevant.
- Do NOT include panic-inducing language.
- Do NOT diagnose or label diseases.
- If something is unclear or missing in the report, clearly say so.
- Do NOT exceed 120 words.

Patient Lab Report:
{state['patient_report']}

Retrieved Medical References (for reference only):
{state['retrieved_docs']}
"""

        response = llm.invoke(prompt)

        return {
            "mtn_output": response.content
        }

    except Exception as e:
        st.error(f"MTN error: {e}")
        return {"mtn_output": ""}


def condition_mapper(state: State) -> State:
    st.info("🧬 Mapping possible conditions...")

    try:
        llm = get_llm()

        prompt = f"""
You are a Medical Report Analyzer.

Task:
- Analyze the lab values in the report.
- Identify patterns and abnormalities.
- Explain what these findings MAY be associated with in general medical terms.

Guidelines:
- Use retrieved medical references as ONE source of truth.
- Use your trained medical knowledge to perform deep reasoning
  and contextual analysis.
- Do NOT diagnose any disease.
- Do NOT assume severity unless clearly indicated.
- Use calm, reassuring language suitable for patients.
- If something is outside the scope of the report, explicitly say so.
- Do NOT exceed 150 words.

Patient Lab Report:
{state['patient_report']}

Retrieved Medical References (for reference only):
{state['retrieved_docs']}
"""

        response = llm.invoke(prompt)

        return {
            "condition_output": response.content
        }

    except Exception as e:
        st.error(f"Condition mapper error: {e}")
        return {"condition_output": ""}


def patient_summary(state: State) -> State:
    st.info("📝 Generating patient-friendly summary...")

    try:
        llm = get_llm()

        prompt = f"""
You are a Medical Report Summarizer for patients.

Task:
- Generate a clear, patient-friendly medical report summary.
- Combine simplified medical terms and analytical insights.
- Help the patient understand the report without causing anxiety.

Tone & Safety Rules:
- Be calm, reassuring, and supportive.
- Avoid alarming or panic-inducing language.
- Do NOT diagnose or predict outcomes.
- Encourage consulting a healthcare professional where appropriate.
- If information is outside the provided medical report, clearly state that.
- Do NOT exceed 180 words.

Simplified Medical Terms:
{state['mtn_output']}

Medical Analysis & Context:
{state['condition_output']}

Final Output:
- Patient-friendly explanation
- What the results generally indicate
- Gentle next steps (non-prescriptive)
"""

        response = llm.invoke(prompt)

        return {
            "patient_summary": response.content
        }

    except Exception as e:
        st.error(f"Summary error: {e}")
        return {"patient_summary": ""}


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
# STREAM OUTPUT
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
uploaded_file = st.file_uploader("📄 Upload Medical Report", type=["pdf"])

if uploaded_file:

    st.success("✅ File uploaded successfully")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("⚙️ Processing your report..."):

        st.write("📖 Reading PDF...")
        docs = load_medical_pdf(temp_path)

        docs = docs[:20]

        st.write("🧹 Cleaning report...")
        text = "\n\n".join([doc.page_content for doc in docs])
        text = text[:8000]

        cleaned = clean_patient_report(text)

        st.write("🔗 Building AI workflow...")
        graph = get_graph()

        st.write("🚀 Running AI Agents...")

        result = graph.invoke({
            "patient_report": cleaned,
            "retrieved_docs": [],
            "mtn_output": "",
            "condition_output": "",
            "patient_summary": ""
        })

    st.subheader("📋 Final Patient Summary")
    stream_output(result.get("patient_summary", ""))

    os.remove(temp_path)