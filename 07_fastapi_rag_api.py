# 07_fastapi_rag_api.py
# FastAPI backend for Resume RAG (Week 7–8)

from pathlib import Path
from typing import List, Any

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------- Config ----------

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db_langchain_resume"  # already created by LangChain script
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt2"


# ---------- Request / Response models ----------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class RagRequest(BaseModel):
    question: str
    top_k: int = 3


class ChunkResult(BaseModel):
    text: str
    distance: float


class RagResponse(BaseModel):
    answer: str
    context: List[ChunkResult]


# ---------- App ----------

app = FastAPI(title="Resume RAG API", version="0.1.0")

# CORS so the Next.js frontend (http://localhost:3000) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],   # includes OPTIONS so preflight does not 405
    allow_headers=["*"],
)

# Global objects initialised at startup
embed_model: SentenceTransformer | None = None
chroma_client: Any = None
chroma_collection: Any = None
llm_pipe: Any = None


# ---------- Startup ----------

@app.on_event("startup")
def on_startup() -> None:
    global embed_model, chroma_client, chroma_collection, llm_pipe

    print("[STARTUP] Initializing embed model + Chroma + LLM...")

    # Embedding model
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print("[EMBED] Model loaded:", EMBED_MODEL_NAME)

    # Chroma client / collection (already persisted from previous scripts)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    chroma_collection = chroma_client.get_or_create_collection("langchain")
    count = chroma_collection.count()
    print(f"[CHROMA] Loaded collection 'langchain' with {count} records.")

    # LLM for answer generation (small local gpt2)
    print("[LLM] Loading gpt2 for chat...")
    llm_pipe = pipeline("text-generation", model=LLM_MODEL_NAME, device=-1)

    print("[STARTUP] Ready. Resume RAG API is initialized.")


# ---------- Helpers ----------

def _search_chunks(query: str, top_k: int = 3) -> List[ChunkResult]:
    if embed_model is None or chroma_collection is None:
        raise RuntimeError("Models not initialized")

    query_emb = embed_model.encode([query]).tolist()
    results = chroma_collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    return [
        ChunkResult(text=text, distance=float(dist))
        for text, dist in zip(docs, dists)
    ]


# ---------- Routes ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    try:
        chunks = _search_chunks(req.query, req.top_k)
        return {"results": [c.model_dump() for c in chunks]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag", response_model=RagResponse)
def answer_rag(req: RagRequest):
    """
    RAG endpoint used by both Swagger and the Next.js frontend.
    """
    try:
        chunks = _search_chunks(req.question, req.top_k)

        # Build context string from top chunks
        context_text = "\n\n".join(c.text for c in chunks)

        # Simple prompt for tiny gpt2 (keep short)
        prompt = (
            "You are reading my resume text and answering a question about it.\n\n"
            f"RESUME CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {req.question}\n"
            "ANSWER (short and clear):"
        )

        raw = llm_pipe(prompt, max_length=256, num_return_sequences=1)[0]["generated_text"]
        answer = raw.split("ANSWER (short and clear):", 1)[-1].strip()

        if not answer:
            answer = chunks[0].text if chunks else "No answer found."

        return RagResponse(
            answer=answer,
            context=chunks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
