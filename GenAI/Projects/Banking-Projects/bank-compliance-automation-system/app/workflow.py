# workflow.py
import os
import uuid
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils import get_embeddings, get_llm, get_documents

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['Langchain_Project'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

def run_workflow(file_path: str, unique_id: str) -> str:
    embeddings = get_embeddings()
    llm = get_llm()

    documents = get_documents(file_path)

    if not documents:
        raise ValueError("No documents found. Please check the file path or format.")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(BASE_DIR, 'faiss_index')
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    uploaded_clauses = [{"text": doc.page_content} for doc in documents]

    prompt_template = PromptTemplate(
        input_variables=["uploaded_clause", "match_1", "match_2"],
        template="""
        You are a compliance assistant. Analyze if the following uploaded clause complies with regulations.

        Uploaded Clause:
        "{uploaded_clause}"

        Top 2 Matching Regulatory Clauses:
        1. "{match_1}"
        2. "{match_2}"

        Think step-by-step:
        - Compare the uploaded clause with the regulations
        - Highlight similarities/differences
        - Return only a structured JSON:
        
        {{"compliant": true/false, "reasoning": "..."}}
    """
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)

    result = []
    for clause in uploaded_clauses:
        query = clause['text']
        matches = vectorstore.similarity_search(query, k=2)
        llm_response = chain.invoke({
            "uploaded_clause": query,
            "match_1": matches[0].page_content,
            "match_2": matches[1].page_content
        })
        result.append({"llm_response": llm_response})

    file_name = f"compliance_{unique_id}.json"
    result_path = os.path.join("results", file_name)

    try:
        with open(result_path, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        raise RuntimeError(f"Failed to save the result file {file_name}: {e}")

    return file_name
