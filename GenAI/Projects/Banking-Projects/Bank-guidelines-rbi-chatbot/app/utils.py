# utils.py
import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# You can read this from environment variables or config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )

@lru_cache()
def get_embeddings():
    """
    Lazily load and cache the OpenAI Embeddings instance.
    Only initialized once per process.
    """
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))