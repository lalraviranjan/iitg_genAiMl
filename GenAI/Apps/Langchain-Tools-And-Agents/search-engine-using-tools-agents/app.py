import streamlit as st
#Arxiv Researcg
# Import Tools Library
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
#Import RAG tools
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
#Convert the retriever into a tool
from langchain.tools.retriever import create_retriever_tool
from langchain.callbacks import StreamlitCallbackHandler

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Duck Duck Go Search")

## Sidebar for settings
# st.sidebar.title("Settings")
# api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

st.title("LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

# Iterate through all the messages stored in the session state
for msg in st.session_state.messages:
    # Display each message in the chat interface with the appropriate role (user or assistant)
    st.chat_message(msg["role"]).write(msg['content'])    

# Check if the user has entered a new input in the chat input box
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Append the user's input to the session state messages with the role "user"
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user's input in the chat interface
    st.chat_message("user").write(prompt)

    # Initialize the language model (LLM) with the specified API key, model name, and streaming enabled
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)
    # Define the tools that the agent can use (DuckDuckGo search, Arxiv, and Wikipedia)
    tools = [search, arxiv, wiki]

    # Initialize the agent with the tools, LLM, and a specific agent type (zero-shot-react-description)
    # Enable error handling for parsing issues
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    # Create a new assistant message in the chat interface
    with st.chat_message("assistant"):
        # Set up a Streamlit callback handler to display the agent's thoughts and actions in real-time
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # Run the agent with the user's messages and the callback handler
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        # Append the assistant's response to the session state messages
        st.session_state.messages.append({'role': 'assistant', "content": response})
        # Display the assistant's response in the chat interface
        st.write(response)