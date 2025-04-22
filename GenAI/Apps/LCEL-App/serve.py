from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define model
model = ChatGroq(api_key=groq_api_key, model="Gemma-2-9b-it")

# Define prompt template properly using from_messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a French language tutor. Translate the following sentence in {language}:"),
    ("user", "{text}")
])

# Define parser
parser = StrOutputParser()

# Create the chain
chain = prompt | model | parser

# Create FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain + Groq"
)

# Add LangServe route
add_routes(app, chain, path="/chain")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
