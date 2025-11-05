# scripts/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scripts.rag_query_gemma import ask_gemma  # we'll define this function in a second

app = FastAPI(title="Gemma Chatbot API")

# --- Allow frontend (React) access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your React URL if you want stricter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define request/response models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def home():
    return {"message": "Gemma Chatbot API running!"}

@app.post("/query", response_model=QueryResponse)
def query_gemma(request: QueryRequest):
    """Handles chatbot queries."""
    answer = ask_gemma(request.question)
    return QueryResponse(answer=answer)
