import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Set upi the Stramlit app
st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Uing Google Gemma 2")

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

## Initializing the tools
# Import and initialize the Wikipedia tool from LangChain.
# This allows the agent to search Wikipedia for factual information on topics.
wikipedia_wrapper=WikipediaAPIWrapper()

# Wrap the Wikipedia function as a Tool so it can be used by the agent.
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"

)

## Initialize the Math tool
# Create a math chain to solve math expressions using the LLM.
math_chain=LLMMathChain.from_llm(llm=llm)

# Define the calculator tool for solving math-related queries.
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to be provided"
)

# Define the prompt to instruct the LLM to act as a logical math-solving agent.
prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
# Create an LLMChain using the above prompt.
# This chain uses the LLM to reason through logic or word-based math problems.
chain=LLMChain(llm=llm,prompt=prompt_template)

# Wrap the reasoning logic as a Tool to be used in the agent setup.
reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)
reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

## Initialize the multi-tool agent
# Create an agent using LangChain's ZERO_SHOT_REACT_DESCRIPTION agent type.
# This allows the agent to choose from the available tools (Wikipedia, Calculator, Reasoning tool) based on the task.
# It will analyze the input and pick the right tool automatically.
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Initialize the chat history if it’s the first session interaction.
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a MAth chatbot who can answer all your maths questions"}
    ]

# Display the chat history (both user and assistant messages) in the Streamlit interface.
# Streamlit internally manages the session state, so messages persist across interactions.
# This allows the user to see the conversation history as they interact with the assistant.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
## LEts start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("find my answer"):
    if question:
        with st.spinner("Thinking..."):
            # Add user message to the session state
            # This stores the user's question in the session state for later reference.
            # The message is appended to the chat history so it can be displayed in the UI.
            # For example, if the user asks a question, it will be stored and displayed in the chat.
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            
            # Set up the callback handler to stream intermediate thoughts (if any) in Streamlit
            # This allows for real-time updates in the Streamlit app as the agent processes the input.
            # The `expand_new_thoughts` parameter controls whether new thoughts are expanded in the UI.
            # StreamlitCallbackHandler - this is streamlit specific callback handler that allows the agent to stream its thoughts to the Streamlit app.
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            
            # Run the agent with the complete message history and callback handler
            # This will process the user question and generate a response using the agent.
            # The response will be appended to the message history for display.
            # The response is generated by the assistant agent, which uses the tools defined above.
            # The assistant agent will analyze the question and decide whether to use the Wikipedia tool, Calculator, or Reasoning tool.
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
                                         )
            st.session_state.messages.append({'role':'assistant',"content":response})
            
            # Display the final response to the user
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")