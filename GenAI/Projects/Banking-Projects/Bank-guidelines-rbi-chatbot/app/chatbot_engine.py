import os
from typing import List
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable import RunnableMap
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.retrievers.multi_query import MultiQueryRetriever

from utils import get_llm, get_embeddings

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_HOST")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# LangChain tracing
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = LANGCHAIN_PROJECT

# === Initialize shared components ===
embeddings = get_embeddings()
llm = get_llm()

# Load persisted vector store
vector_db = Chroma(
    persist_directory="indian_banking_fin_report_index",
    embedding_function=embeddings
)

# MultiQuery retriever using LLM
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db.as_retriever(),
    llm=llm,
    include_original=True
)

# === Table query logic ===
def is_table_query(query: str) -> bool:
    keywords = ["table", "figure", "amount", "data", "statistics", "reserves", "value", "year", "percentage", "increase", "decrease"]
    return any(k.lower() in query.lower() for k in keywords)

def get_dynamic_retriever(query: str):
    if is_table_query(query):
        return vector_db.as_retriever(search_kwargs={"filter": {"source": "table"}})
    return vector_db.as_retriever()

# === Prompt ===
chat_prompt = ChatPromptTemplate.from_messages([
   ("system", """You are an expert assistant trained on the RBI Annual Report 2023-24.
    Respond precisely and clearly using the context from the report.
    - If the question involves numerical or tabular data (like figures, statistics, or financials), answer with exact values in markdown tables where applicable.
    - When explaining in points, use concise bullet points that highlight key ideas. Use icons for each point to enhance clarity and engagement.
    - Use icons to enhance clarity and engagement.
    - Never mention the page or table number of the report.
    - If the context does not provide an answer, reply with: "The report does not specify this."
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

parser = StrOutputParser()

# === Chain ===
context_chain = RunnableMap({
    "input": lambda x: x["input"],
    "chat_history": lambda x: x.get("chat_history", []),
    "context": lambda x: "\n\n".join(
        doc.page_content for doc in multi_query_retriever.invoke(x["input"])
    )
}) | chat_prompt | llm | parser

# === Redis-based history ===
def get_redis_history(session_id: str):
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

runnable = RunnableWithMessageHistory(
    context_chain,
    get_redis_history,
    input_message_key="input",
    history_messages_key="chat_history"
)

# === Chat entry point ===
def get_chat_response(user_input: str, session_id: str) -> str:
    try:
        return runnable.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
    except Exception as e:
        return f"An error occurred: {str(e)}"
