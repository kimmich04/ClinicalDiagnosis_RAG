from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ---- CONFIG ----
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
EMBED_MODEL = "openai/clip-vit-base-patch32"
GEMMA_MODEL = "google/gemma-3-4b-it"   # you can change to gemma-3-4b-it if GPU allows
TOP_K = 5

# ---- SETUP ----
app = FastAPI()

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your React app URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
print("Loading models...")
clip_model = CLIPModel.from_pretrained(EMBED_MODEL)
clip_proc = CLIPProcessor.from_pretrained(EMBED_MODEL)
qdrant = QdrantClient(url=QDRANT_URL)
tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL)
gemma_model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL, device_map="auto", torch_dtype="auto")
gemma_pipeline = pipeline("text-generation", model=gemma_model, tokenizer=tokenizer)

# ---- DATA MODEL ----
class QueryRequest(BaseModel):
    query: str

# ---- FUNCTIONS ----
def get_text_embedding(text: str):
    inputs = clip_proc(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy().tolist()

def retrieve_context(query_vec):
    results = qdrant.query_points(collection_name=COLLECTION_NAME, query=query_vec, limit=TOP_K)
    chunks = [p.payload["text"] for p in results.points if p.payload.get("type") == "text"]
    return "\n\n".join(chunks)

# ---- ROUTE ----
@app.post("/ask")
def ask_question(data: QueryRequest):
    query = data.query
    query_vec = get_text_embedding(query)
    context = retrieve_context(query_vec)

    prompt = (
        f"Answer the following question using the given medical record context:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    response = gemma_pipeline(prompt, max_new_tokens=300, do_sample=False)
    answer = response[0]["generated_text"].split("Answer:")[-1].strip()

    return {"answer": answer}
