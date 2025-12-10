"""
09_customer_support_rag.py

Month 3 – Project: Customer Support AI (RAG)

Goal:
- Load simple "support knowledge base" (FAQ / help text).
- Chunk + embed the text.
- Store in a ChromaDB collection.
- Expose FastAPI endpoints for:
    - /support-health
    - /support-search
    - /support-rag

This is separate from the resume RAG API (07_fastapi_rag_api.py).
"""

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# -------------------- Paths & constants --------------------

BASE_DIR = Path(__file__).resolve().parent

# You can edit this file with your own FAQ / support docs
SUPPORT_KB_PATH = BASE_DIR / "data" / "support_faq.txt"

CHROMA_DIR_SUPPORT = BASE_DIR / "chroma_db_support"
CHROMA_COLLECTION_NAME_SUPPORT = "support_kb"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------- Request models --------------------

class SupportSearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SupportRagRequest(BaseModel):
    question: str
    top_k: int = 3


# -------------------- Globals --------------------

app = FastAPI(title="Customer Support RAG API")

embed_model: SentenceTransformer | None = None
chroma_client_support = None
chroma_collection_support = None


# -------------------- Utilities --------------------

def ensure_support_kb_file():
    """
    Ensure data/support_faq.txt exists.
    If not, create a small demo FAQ file so the API still works.
    """
    if not SUPPORT_KB_PATH.exists():
        SUPPORT_KB_PATH.parent.mkdir(parents=True, exist_ok=True)
        demo_text = """FAQ: FLEX FOAM B2B Platform

Q: What is FLEX FOAM?
A: FLEX FOAM is a B2B e-commerce SaaS platform for foam manufacturers and distributors.

Q: What can I do on the platform?
A: You can manage foam products, bulk orders, distributors, invoices, and payments.

Q: Who is the typical user?
A: Factory admins, sales reps, distributors, and B2B customers who buy foam products in bulk.

Q: How does pricing work?
A: Pricing supports different tiers, volume discounts, and custom quotes for large orders.

Q: How are payments handled?
A: The platform integrates with Paystack for NGN payments and can be extended to other gateways.

Q: How can I contact support?
A: You can email support@flexfoam.example or use the in-app support chat.
"""
        SUPPORT_KB_PATH.write_text(demo_text, encoding="utf-8")
        print(f"[SUPPORT KB] Created demo support_faq.txt at: {SUPPORT_KB_PATH}")
    else:
        print(f"[SUPPORT KB] Using existing file at: {SUPPORT_KB_PATH}")


def load_support_kb_text() -> str:
    ensure_support_kb_file()
    text = SUPPORT_KB_PATH.read_text(encoding="utf-8")
    return text


def simple_paragraph_split(text: str) -> List[str]:
    """
    Split the KB text into paragraphs separated by blank lines.
    Each paragraph will be treated as a 'chunk' / doc.
    """
    # Normalize line endings
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on double newlines
    raw_chunks = normalized.split("\n\n")
    chunks: list[str] = []

    for raw in raw_chunks:
        c = raw.strip()
        if not c:
            continue
        chunks.append(c)

    return chunks


def init_embed_model_support():
    global embed_model
    if embed_model is None:
        print("[EMBED] Loading SentenceTransformer model for support KB:", EMBED_MODEL_NAME)
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def init_chroma_support():
    """
    Initialize ChromaDB for support KB. If empty, build from support_faq.txt.
    """
    global chroma_client_support, chroma_collection_support

    if chroma_client_support is None:
        print("[CHROMA][SUPPORT] Initializing persistent client at:", CHROMA_DIR_SUPPORT)
        chroma_client_support = chromadb.PersistentClient(
            path=str(CHROMA_DIR_SUPPORT),
            settings=Settings(allow_reset=True),
        )

    try:
        chroma_collection_support = chroma_client_support.get_collection(CHROMA_COLLECTION_NAME_SUPPORT)
        print("[CHROMA][SUPPORT] Loaded existing collection:", CHROMA_COLLECTION_NAME_SUPPORT)
    except Exception:
        print("[CHROMA][SUPPORT] Creating new collection:", CHROMA_COLLECTION_NAME_SUPPORT)
        chroma_collection_support = chroma_client_support.create_collection(CHROMA_COLLECTION_NAME_SUPPORT)

    count = chroma_collection_support.count()
    print(f"[CHROMA][SUPPORT] Existing collection has {count} records.")

    if count == 0:
        print("[CHROMA][SUPPORT] Building collection from support KB file...")
        text = load_support_kb_text()
        paragraphs = simple_paragraph_split(text)
        print(f"[SUPPORT KB] Number of paragraphs: {len(paragraphs)}")

        if not paragraphs:
            raise RuntimeError("Support KB has no paragraphs to index.")

        init_embed_model_support()
        embeddings = embed_model.encode(paragraphs, convert_to_numpy=True)

        ids = [f"support-{i}" for i in range(len(paragraphs))]
        chroma_collection_support.add(
            ids=ids,
            documents=paragraphs,
            embeddings=embeddings.tolist(),
        )
        print("[CHROMA][SUPPORT] Support KB collection built.")


def support_search_internal(query: str, top_k: int = 3):
    if chroma_collection_support is None:
        raise RuntimeError("Support Chroma collection not initialized")

    init_embed_model_support()
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0].tolist()

    results = chroma_collection_support.query(
        query_embeddings=[q_vec],
        n_results=top_k,
    )
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return docs, distances


# -------------------- FastAPI lifecycle --------------------

@app.on_event("startup")
def on_startup():
    print("[STARTUP][SUPPORT] Initializing support KB RAG API...")
    init_embed_model_support()
    init_chroma_support()
    print("[STARTUP][SUPPORT] Ready.")


# -------------------- Endpoints --------------------

@app.get("/support-health")
def support_health():
    return {
        "status": "ok",
        "detail": "Customer Support RAG API is running",
    }


@app.post("/support-search")
def support_search(payload: SupportSearchRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    docs, distances = support_search_internal(payload.query, top_k=payload.top_k)
    results = []
    for doc, dist in zip(docs, distances):
        results.append({"text": doc, "distance": float(dist)})

    return {"results": results}


@app.post("/support-rag")
def support_rag(payload: SupportRagRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    docs, distances = support_search_internal(payload.question, top_k=payload.top_k)
    if not docs:
        return {"answer": "No relevant information found in the support knowledge base.", "context": []}

    best_doc = docs[0]
    ctx = []
    for doc, dist in zip(docs, distances):
        ctx.append({"text": doc, "distance": float(dist)})

    return {
        "answer": best_doc,
        "context": ctx,
    }


if __name__ == "__main__":
    # Optional: allow running with "python 09_customer_support_rag.py"
    import uvicorn

    uvicorn.run("09_customer_support_rag:app", host="127.0.0.1", port=8001, reload=True)
