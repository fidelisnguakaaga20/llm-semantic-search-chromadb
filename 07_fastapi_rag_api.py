"""
07_fastapi_rag_api.py

Week 7 of the LLM roadmap:
FastAPI backend with routes:

- GET  /health
- POST /embed   -> return embeddings for given texts
- POST /search  -> semantic search over resume chunks
- POST /rag     -> simple RAG answer (best chunk)
- POST /chat    -> free-form chat using gpt2 (stateless)
- POST /agent   -> simple "agent" that decides when to use resume search

We reuse the existing vector DB created by 06_langchain_resume_rag.py:

- chroma_db_langchain_resume/   (persistent Chroma directory)
- collection name: "langchain"
"""

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from transformers import pipeline

# Force HuggingFace to run OFFLINE (use local cache only)
os.environ["HF_HUB_OFFLINE"] = "1"


# -------------------- Paths & constants --------------------

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "sample.pdf"  # only for reference if needed

# IMPORTANT: reuse the LangChain Chroma DB already built
CHROMA_DIR = BASE_DIR / "chroma_db_langchain_resume"
CHROMA_COLLECTION_NAME = "langchain"  # LangChain default

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------- Models for requests --------------------

class EmbedRequest(BaseModel):
    texts: List[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class RagRequest(BaseModel):
    question: str
    top_k: int = 3


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class AgentRequest(BaseModel):
    question: str


# -------------------- Global objects (loaded once) --------------------

app = FastAPI(title="Resume RAG API")

embed_model: SentenceTransformer | None = None
chroma_client = None
chroma_collection = None
chat_llm = None  # gpt2 pipeline


# -------------------- Utility functions --------------------

def load_pdf_text(pdf_path: Path) -> str:
    """Not used in the API logic, kept for reference."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    reader = PdfReader(str(pdf_path))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    full_text = "\n".join(text_parts)
    return full_text


def init_embed_model():
    global embed_model
    if embed_model is None:
        print("[EMBED] Loading SentenceTransformer model (offline):", EMBED_MODEL_NAME)
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def init_chroma():
    """
    Initialize Chroma client + collection by reusing the existing DB
    built earlier by 06_langchain_resume_rag.py.
    """
    global chroma_client, chroma_collection

    if chroma_client is None:
        print("[CHROMA] Initializing persistent client at:", CHROMA_DIR)
        chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(allow_reset=True),
        )

    try:
        chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        count = chroma_collection.count()
        print(f"[CHROMA] Loaded collection '{CHROMA_COLLECTION_NAME}' with {count} records.")
        if count == 0:
            print(
                "[CHROMA] WARNING: collection is empty. "
                "Run 06_langchain_resume_rag.py once to build it."
            )
    except Exception as e:
        raise RuntimeError(
            f"[CHROMA] Could not load collection '{CHROMA_COLLECTION_NAME}' "
            f"from {CHROMA_DIR}. Make sure 06_langchain_resume_rag.py has been run."
        ) from e


def init_chat_llm():
    global chat_llm
    if chat_llm is None:
        print("[LLM] Loading gpt2 for chat...")
        chat_llm = pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )


def resume_search(query: str, top_k: int = 3):
    """Semantic search over resume chunks."""
    if chroma_collection is None:
        raise RuntimeError("Chroma collection not initialized")

    init_embed_model()
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0].tolist()
    results = chroma_collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
    )
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return docs, distances


def should_use_resume_tool(question: str) -> bool:
    q = question.lower()
    keywords = [
        "resume",
        "cv",
        "my skills",
        "my experience",
        "my projects",
        "my education",
        "according to my resume",
    ]
    return any(k in q for k in keywords)


# -------------------- FastAPI lifecycle --------------------

@app.on_event("startup")
def on_startup():
    print("[STARTUP] Initializing embed model + Chroma + LLM...")
    init_embed_model()
    init_chroma()
    init_chat_llm()
    print("[STARTUP] Ready. Resume RAG API is initialized.")


# -------------------- Endpoints --------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "detail": "Resume RAG API running"}


@app.post("/embed")
def embed_texts(payload: EmbedRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")

    init_embed_model()
    vectors = embed_model.encode(payload.texts, convert_to_numpy=True)
    return {"embeddings": vectors.tolist()}


@app.post("/search")
def semantic_search(payload: SearchRequest):
    docs, distances = resume_search(payload.query, top_k=payload.top_k)
    results = []
    for doc, dist in zip(docs, distances):
        results.append({"text": doc, "distance": float(dist)})
    return {"results": results}


@app.post("/rag")
def rag_answer(payload: RagRequest):
    docs, distances = resume_search(payload.question, top_k=payload.top_k)
    if not docs:
        return {"answer": "No information found in resume.", "context": []}

    best_doc = docs[0]
    ctx = []
    for doc, dist in zip(docs, distances):
        ctx.append({"text": doc, "distance": float(dist)})

    return {
        "answer": best_doc,
        "context": ctx,
    }


@app.post("/chat")
def chat(payload: ChatRequest):
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty")

    init_chat_llm()

    conv_text = ""
    for m in payload.messages:
        role = m.role.upper()
        conv_text += f"{role}: {m.content}\n"
    conv_text += "ASSISTANT:"

    out = chat_llm(conv_text)[0]["generated_text"]
    last_idx = out.rfind("ASSISTANT:")
    if last_idx != -1:
        answer = out[last_idx + len("ASSISTANT:"):].strip()
    else:
        answer = out.strip()

    return {"answer": answer}


@app.post("/agent")
def agent(payload: AgentRequest):
    """
    Simple "agent":

    - If question is about resume/skills, call resume_search TOOL.
    - Otherwise, just do a free-form chat completion.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    if should_use_resume_tool(question):
        docs, distances = resume_search(question, top_k=3)
        if not docs:
            tool_answer = "I could not find anything in your resume related to that question."
        else:
            tool_answer = (
                "Based on your resume, here are the most relevant parts:\n\n"
                + "\n\n".join(docs)
            )
        return {
            "mode": "resume_tool",
            "answer": tool_answer,
        }

    init_chat_llm()
    prompt = (
        "You are helping Nguakaaga think about their LLM engineering career.\n\n"
        f"USER: {question}\n\nASSISTANT:"
    )
    out = chat_llm(prompt)[0]["generated_text"]
    last_idx = out.rfind("ASSISTANT:")
    if last_idx != -1:
        answer = out[last_idx + len("ASSISTANT:"):].strip()
    else:
        answer = out.strip()

    return {
        "mode": "chat",
        "answer": answer,
    }
