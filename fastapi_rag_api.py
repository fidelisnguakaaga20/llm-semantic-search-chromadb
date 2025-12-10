"""
FastAPI backend for resume RAG + utility endpoints (LOCAL HF VERSION).

- Uses SentenceTransformers for embeddings (no OpenAI).
- Uses a small HuggingFace model (distilgpt2) for generation (no OpenAI).
- Stores resume chunks + embeddings in memory.
- Endpoints:
  - GET  /health
  - POST /upload-resume
  - POST /embed
  - POST /search
  - POST /rag
  - POST /rag-stream   (streaming answers, chunked by us)
  - POST /chat
  - POST /agent
"""

import os
import math
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "distilgpt2"

RESUME_PDF_PATH = os.path.join("data", "sample.pdf")

# In-memory store for resume chunks
resume_chunks: List[dict] = []


# ---------------------------------------------------------
# LOAD LOCAL MODELS (ONCE)
# ---------------------------------------------------------

print("Loading embedding model:", EMBED_MODEL_NAME)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading generation model:", GEN_MODEL_NAME)
text_gen = pipeline("text-generation", model=GEN_MODEL_NAME)


# ---------------------------------------------------------
# FASTAPI APP + CORS
# ---------------------------------------------------------

app = FastAPI(title="Resume RAG API (Local HF)")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Pydantic MODELS
# ---------------------------------------------------------

class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    document: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class RagRequest(BaseModel):
    query: str
    k: int = 5
    max_new_tokens: int = 256


class RagChunk(BaseModel):
    text: str
    score: float


class RagResponse(BaseModel):
    answer: str
    chunks: List[RagChunk]


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 256
    system_prompt: Optional[str] = (
        "You are a helpful AI assistant. Answer clearly and concisely."
    )


class ChatResponse(BaseModel):
    reply: str


class AgentRequest(BaseModel):
    message: str
    mode: Literal["auto", "resume", "chat"] = "auto"
    max_new_tokens: int = 256


class AgentResponse(BaseModel):
    reply: str
    used_tool: Literal["rag", "chat"]


class UploadResumeResponse(BaseModel):
    status: str
    chunks_indexed: int


# ---------------------------------------------------------
# UTILS: EMBEDDINGS + COSINE SIM
# ---------------------------------------------------------

def embed_text(text: str) -> List[float]:
    # SentenceTransformers returns numpy array; convert to list
    vec = embed_model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------
# UTILS: BUILD CHUNKS FROM PDF
# ---------------------------------------------------------

def _chunk_text_for_index(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def rebuild_chunks_from_pdf(pdf_path: str) -> int:
    """
    Read resume PDF from disk, build text chunks, compute embeddings,
    and store in the global resume_chunks list.
    """
    global resume_chunks

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        parts.append((page.extract_text() or ""))

    full_text = "\n".join(parts).strip()
    if not full_text:
        raise ValueError("Could not extract any text from PDF.")

    texts = _chunk_text_for_index(full_text, chunk_size=800, overlap=200)
    if not texts:
        raise ValueError("No chunks created from PDF text.")

    new_chunks = []
    for t in texts:
        emb = embed_text(t)
        new_chunks.append({"text": t, "embedding": emb})

    resume_chunks = new_chunks
    print(f"Rebuilt resume chunks: {len(resume_chunks)}")
    return len(resume_chunks)


def ensure_resume_chunks() -> None:
    """
    Ensure resume_chunks is populated.
    If empty but PDF exists, rebuild automatically.
    """
    global resume_chunks

    if resume_chunks:
        return

    if not os.path.exists(RESUME_PDF_PATH):
        raise HTTPException(
            status_code=404,
            detail="No resume chunks indexed yet and no PDF found on disk. "
                   "Upload a resume via /upload-resume.",
        )

    try:
        count = rebuild_chunks_from_pdf(RESUME_PDF_PATH)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild resume chunks from disk: {e}",
        )
    if count == 0:
        raise HTTPException(
            status_code=500,
            detail="Rebuilt resume, but no chunks were created.",
        )


# ---------------------------------------------------------
# UTILS: LOCAL GENERATION
# ---------------------------------------------------------

def generate_with_context(system: str, user_msg: str, max_new_tokens: int = 256) -> str:
    """
    Use a local small language model (distilgpt2) to generate an answer.
    We approximate length control using max_length.
    """
    prompt = f"{system}\n\n{user_msg}\n\nAnswer:"
    # Rough length control in tokens (approximate via words)
    max_length = min(len(prompt.split()) + max_new_tokens, 256)

    outputs = text_gen(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=False,
    )
    text = outputs[0]["generated_text"]
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()


def chunk_text_for_stream(text: str, chunk_size: int = 50):
    """
    Yield small pieces of text for streaming.
    """
    for i in range(0, len(text), chunk_size):
        yield text[i: i + chunk_size]


# ---------------------------------------------------------
# UTILS: RAG
# ---------------------------------------------------------

def run_rag(query: str, k: int, max_new_tokens: int) -> RagResponse:
    ensure_resume_chunks()

    query_emb = embed_text(query)

    scored = []
    for ch in resume_chunks:
        score = cosine_similarity(query_emb, ch["embedding"])
        scored.append((score, ch["text"]))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:k]

    context_lines = []
    chunks_out: List[RagChunk] = []
    for score, text in top:
        context_lines.append(text)
        chunks_out.append(RagChunk(text=text, score=float(score)))

    context = "\n\n".join(context_lines)

    system = (
        "You are an assistant that answers questions strictly based on the user's "
        "resume context. If the answer is not in the context, say you don't know."
    )
    user_block = (
        f"RESUME CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer clearly and concisely."
    )

    answer = generate_with_context(system, user_block, max_new_tokens=max_new_tokens)
    return RagResponse(answer=answer, chunks=chunks_out)


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload-resume", response_model=UploadResumeResponse)
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(os.path.dirname(RESUME_PDF_PATH), exist_ok=True)

    contents = await file.read()
    with open(RESUME_PDF_PATH, "wb") as f:
        f.write(contents)

    try:
        count = rebuild_chunks_from_pdf(RESUME_PDF_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return UploadResumeResponse(status="ok", chunks_indexed=count)


@app.post("/embed", response_model=EmbedResponse)
def embed_endpoint(payload: EmbedRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    emb = embed_text(payload.text)
    return EmbedResponse(embedding=emb, dim=len(emb))


@app.post("/search", response_model=SearchResponse)
def search_endpoint(payload: SearchRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    ensure_resume_chunks()

    query_emb = embed_text(payload.query)

    scored = []
    for ch in resume_chunks:
        score = cosine_similarity(query_emb, ch["embedding"])
        scored.append((score, ch["text"]))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[: payload.k]

    results: List[SearchResult] = [
        SearchResult(document=text, score=float(score)) for score, text in top
    ]
    return SearchResponse(results=results)


@app.post("/rag", response_model=RagResponse)
def rag_endpoint(payload: RagRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    return run_rag(
        query=payload.query,
        k=payload.k,
        max_new_tokens=payload.max_new_tokens,
    )


@app.post("/rag-stream")
def rag_stream_endpoint(payload: RagRequest):
    """
    Streaming version of /rag.
    We run RAG once to get the full answer text, then stream it in chunks.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    rag_result = run_rag(
        query=payload.query,
        k=payload.k,
        max_new_tokens=payload.max_new_tokens,
    )
    answer_text = rag_result.answer

    def generate():
        for piece in chunk_text_for_stream(answer_text, chunk_size=60):
            yield piece

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    system = payload.system_prompt or "You are a helpful AI assistant."
    reply = generate_with_context(system, payload.message, max_new_tokens=payload.max_new_tokens)
    return ChatResponse(reply=reply)


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(payload: AgentRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    if payload.mode == "resume":
        chosen = "rag"
    elif payload.mode == "chat":
        chosen = "chat"
    else:
        text_lower = payload.message.lower()
        resume_keywords = [
            "resume",
            "cv",
            "experience",
            "skills",
            "projects",
            "job history",
            "education",
        ]
        chosen = "rag" if any(kw in text_lower for kw in resume_keywords) else "chat"

    if chosen == "rag":
        rag_result = run_rag(
            query=payload.message,
            k=5,
            max_new_tokens=payload.max_new_tokens,
        )
        reply = rag_result.answer
    else:
        system = "You are a helpful AI assistant. Answer clearly and concisely."
        reply = generate_with_context(system, payload.message, max_new_tokens=payload.max_new_tokens)

    return AgentResponse(reply=reply, used_tool=chosen)


# To run locally:
#   cd llm-roadmap
#   uvicorn fastapi_rag_api:app --reload --port 8000
