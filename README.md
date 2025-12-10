# Resume RAG Backend (FastAPI + Chroma + GPT-2)

Backend for a Resume RAG chatbot.

## Stack

- FastAPI
- SentenceTransformers (`all-MiniLM-L6-v2`)
- Chroma DB (persistent folder)
- GPT-2 via HuggingFace Transformers
- LangChain for PDF loading & chunking

## Endpoints

- `GET /health`
- `POST /upload-resume`
- `POST /embed`
- `POST /search`
- `POST /rag`
- `POST /chat`
- `POST /agent`

## Run locally

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash

pip install -r requirements.txt
uvicorn 07_fastapi_rag_api:app --reload

