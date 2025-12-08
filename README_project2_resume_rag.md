# Project 2 – Resume RAG API (FastAPI + ChromaDB)

Small backend API that performs semantic search and simple RAG over my **resume PDF**.

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- SentenceTransformers – `sentence-transformers/all-MiniLM-L6-v2`
- ChromaDB (local persistent DB)
- LangChain (for earlier CLI RAG script)
- PDF parsing: `pypdf`

## Files

- `data/sample.pdf` – resume used as RAG source
- `data/05_rag_pdf_basic.py` – basic PDF → chunks → Chroma → CLI Q&A
- `06_langchain_resume_rag.py` – LangChain RAG over resume (terminal)
- `07_fastapi_resume_api.py` – FastAPI app exposing RAG endpoints

## How to Run

```bash
cd C:\Users\HP\Desktop\llm-roadmap
.\.venv\Scripts\activate

# Start the FastAPI server
uvicorn 07_fastapi_resume_api:app --reload



Then open:

Swagger UI: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

Endpoints
POST /search

Semantic search over resume chunks.

Request body

{
  "query": "What tech stack do I use in my projects?",
  "k": 3
}


Response (shape)

{
  "query": "What tech stack do I use in my projects?",
  "results": [
    { "text": "...chunk text...", "distance": 1.23 },
    { "text": "...", "distance": 1.45 }
  ]
}

POST /rag

Simple RAG: retrieve and return the best chunk as the “answer”.

Request body

{
  "question": "What tech stack do I use in my projects?",
  "k": 3
}


Response (shape)

{
  "question": "What tech stack do I use in my projects?",
  "answer": "...best chunk text...",
  "context": [
    { "text": "...chunk 1...", "distance": 1.23 },
    { "text": "...chunk 2...", "distance": 1.45 }
  ]
}
