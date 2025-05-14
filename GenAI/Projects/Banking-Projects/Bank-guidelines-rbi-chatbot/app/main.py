from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from chatbot_engine import get_chat_response

app = FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    session_id: str
    user_input: str

@app.get("/")
def read_root():
    return {"message": "Chatbot API is running."}

@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    response = get_chat_response(request.user_input, request.session_id)
    return {"response": response}
