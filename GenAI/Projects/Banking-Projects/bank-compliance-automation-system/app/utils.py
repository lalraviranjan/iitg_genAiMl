# utils.py
import os
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader

@lru_cache(maxsize=1)
def get_llm():
    groq_api_key = os.getenv('GROQ_API_KEY')
    return ChatGroq(api_key=groq_api_key, model="Gemma2-9b-it")

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name='BAAI/bge-large-en-v1.5',
        model_kwargs={"device": "cuda"},
        encode_kwargs={'normalize_embeddings': False}
    )

def load_file(file_path: str):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyMuPDFLoader(file_path)
    elif ext == 'docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

def get_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    documents = load_file(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
