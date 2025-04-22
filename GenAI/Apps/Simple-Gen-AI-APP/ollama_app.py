import os
import time
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# Set LangChain environment configurations
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Create our Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Please respond to the question asked."),
        ("user", "{question}")
    ]
)

# Streamlit framework UI
st.title("Langchain with LLaMA3")
input_text = st.text_input("What question do you have in mind?")

# LLaMA3 model using Ollama
llm = OllamaLLM(model="llama3")
output_parser = StrOutputParser()

# Create the chain: prompt → LLM → parser
chain = prompt | llm | output_parser

# Placeholder for response
response_placeholder = st.empty()
thinking_placeholder = st.empty()

# Trigger chain on user input
if input_text:
    # Clear previous response
    response_placeholder.empty()
    thinking_placeholder.empty()

    thinking_text = "Thinking"

    # Show Thinking... with animation
    for i in range(20): 
        for dot in range(1, 5):
            thinking_placeholder.markdown(f"{thinking_text}{'.' * dot}")
            # Delay between dots
            time.sleep(0.5) 
    
    # Invoke the chain with the user's input question to get the AI-generated response
    response = chain.invoke({"question": input_text})
    
    # Clear "Thinking..." text and start typing response
    thinking_placeholder.empty()

    # st.write(response)
    display_text = ""
    for char in response:
        display_text += char
        response_placeholder.markdown(display_text)
        time.sleep(0.01)