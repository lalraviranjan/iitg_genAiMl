# RAG Q&A Chatbot with PDF uploads
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from chromadb.config import Settings  # Make sure chromadb is installed
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#Input GROQ API key
llm=ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
session_id = "default"

# Cache the model to avoid reloading them every time
@st.cache_resource
def load_llm():
    return ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

# Display loading spinner while loading
with st.spinner("Loading Model... This may take a while."):
    llm = load_llm()

@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# Display loading spinner while loading
with st.spinner("Loading embeddings... This may take a while."):
    embeddings = load_embeddings()
    
# st.success("Model loaded successfully!")

# setup streamlist app
st.title("Conversational RAG with PDF uploads and History")
st.write("Upload a PDF and chat with its context.")


chroma_settings = Settings(
    persist_directory="./chroma_db"  # Directory to store the database
)

#statefully manage chat history
store={}

if 'store' not in st.session_state:
    st.session_state.store = {}

# Upload and Process PDF file 
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)
if uploaded_file:
    documents = []
    temppdf = f"./temp.pdf"
    with open(temppdf, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name
        
    # `temppdf` is the path to the PDF file; `loader.load()` returns the document as a list of LangChain Document objects.
    loader = PyPDFLoader(temppdf)
   
    # These documents are then added to the `documents` list which will later be used for chunking and vector storage.
    docs = loader.load()
    documents.extend(docs)

    # Using RecursiveCharacterTextSplitter to break large documents into smaller, overlapping chunks.
    # This is essential because LLMs have a token limit, and splitting ensures each chunk is within that limit.
    # Overlapping chunks (here, 500 tokens) help maintain context continuity between chunks,
    # which improves the quality of retrieval and the final LLM response during the RAG process.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    
    # Creating a Chroma vector store from the split documents using the specified embedding model.
    # This step converts each document chunk into a high-dimensional vector using embeddings,
    # and stores them in Chroma — a lightweight, persistent vector database optimized for similarity search.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        client_settings=chroma_settings
    )
    
    # Converting the vector store into a retriever object which can be used to fetch relevant document chunks.
    # Setting `k=3` means that for any given user query, the top 3 most similar chunks (based on vector similarity) will be retrieved.
    # This retriever will be plugged into the RAG pipeline to provide contextual grounding to the LLM during generation.
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    # System prompt for rephrasing user questions into standalone queries using prior chat history.
    contexutalize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history, Do not answer the question"
        "just reformulate it if needed and otherwise return as is."
    )

    # Defining a chat prompt using the above system instruction, along with placeholders for past messages and new user input.
    # This template is used by the LLM to generate a clarified, standalone version of the question.
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contexutalize_q_system_prompt),
            MessagesPlaceholder("chat_history"),  # Dynamic placeholder for previous conversation turns
            ("human", "{input}"),  # Placeholder for the latest user input
        ]
    )

    # Creating a retriever that understands conversation history.
    # It uses the standalone question reformulated by the LLM to fetch the most relevant chunks from the vector DB.
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # System prompt that defines how the assistant should behave during the final answer generation.
    # It instructs the assistant to rely only on the retrieved context, keep responses short, and admit when unsure.
    system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use five sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}" # Retrieved documents will be injected here
            )

    # Defining the question-answering prompt template.
    # It combines system instructions, chat history, and the latest user question.
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # Building a "stuff" chain — one of LangChain's document combination strategies.
    # This chain takes all retrieved documents, 'stuffs' them into the prompt, and asks the LLM to answer the query.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Final RAG chain: combines the history-aware retriever with the QA chain.
    # This pipeline first reformulates the question, fetches relevant documents, and then generates an answer based on context.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # `get_session_history` is a function that returns a session-specific message history object (e.g., from Redis, in-memory, etc.).
    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    # Wrapping the RAG chain with session-based memory using RunnableWithMessageHistory.
    # This enables the chain to maintain and utilize chat history across turns (multi-turn conversations).
    # `history_messages_key` is the key used in the prompt templates to inject past chat messages (like "chat_history").
    # `output_messages_key` is the key where the model's answer will be saved (e.g., "answer"), to keep the conversation state updated.
    # This setup allows the GenAI assistant to feel more like a chatbot that remembers what was said earlier.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, 
        get_session_history=get_session_history,
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    user_input = st.text_input("Your Question:")
    if user_input:
        session_history = get_session_history(session_id)
        # Invoking the conversational RAG chain with the user's input and session configuration.
        # `{"input": user_input}` passes the latest user query into the chain.
        # `config={"configurable": {"session_id": session_id}}` ensures that the correct chat history is used
        # by tying the interaction to a specific session (enabling memory across messages).
        # The RAG pipeline will: 
        # - Rephrase the question if needed using chat history,
        # - Retrieve relevant documents from the vector store,
        # - Generate a context-aware response using the LLM.
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable":{"session_id":session_id}
            },
        )
        # st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        # st.write("Chat History:", session_history.messages)
        

