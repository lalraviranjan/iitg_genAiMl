import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# config = {
#     "configurable":{"session_id":"chat_1"}
# }

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "state" not in st.session_state:
    st.session_state.state = {}
    
store= st.session_state.state

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        history = ChatMessageHistory()
        history.add_message(
            SystemMessage(content="""
                          You are an expert assistant in AI / ML. Answer all the technical queries 
                          related to AI and ML domain only.
                          Answer precisely and simple to understand in not more than 40-50 words. Add 1 or 2 pointers in your response if needed.
                          If any queries beyond the tech domain answer gracefully
                          framing similar answers like: I am AIML tutor and this is outside my domain knowledge.
                          """)
        )
        store[session_id] = history
    return store[session_id]

chat_with_msg_history = RunnableWithMessageHistory(llm, get_session_history)

def invoke_chat(instruction):
    chat_response = chat_with_msg_history.invoke(
        [
            HumanMessage(content=f"{instruction}")
        ],
        config={"configurable":{"session_id":"chat_1"}}
    )
    return chat_response.content

st.title("Gen AI Chatbot")
user_input = st.text_input("Enter your query here...")
if user_input:
    answer = invoke_chat(user_input)
    st.session_state.messages.insert(0, {
        'human':user_input,
        'ai':answer
    })

for msg in st.session_state.messages:
    st.markdown(f"**You:** {msg['human']}")
    st.markdown(f"**AI:** {msg['ai']}")