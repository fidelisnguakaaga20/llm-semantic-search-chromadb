Resume RAG API — Project Documentation

By Nguakaaga Mvendaga
LLM Engineering Roadmap — Project 2

🚀 Overview

This project implements a full Retrieval-Augmented Generation backend that processes a resume PDF, stores chunks inside a vector database, and exposes an API for:

Embeddings

Semantic search

RAG answers

Chat (GPT-2 local)

Agent with tool-calling logic

It follows Weeks 5–7 of the Master LLM Engineering Roadmap.

🧠 Features
✅ PDF Processing

PDF text extraction using pypdf

Chunking with overlap for better retrieval

✅ Embeddings

SentenceTransformer: all-MiniLM-L6-v2

Offline loading fallback

Generates resume/skill embeddings

✅ Vector Database (ChromaDB)

Persistent DB on disk

Stores every resume chunk

Fast similarity search

✅ RAG Pipeline

Given a question:

Embed the query

Search the resume vector DB

Return best chunk + full context

✅ Chat

Uses GPT-2 pipeline

Full chat role formatting

Stateless conversation

✅ Agent

Decides whether to:

Use the resume search tool

Or answer normally using GPT-2

Triggers when user asks:

“my resume”,

“my experience”,

“my skills”, etc.

🏗 API Endpoints
GET /health

Check if API is running.

POST /embed

Generate embeddings for a list of texts.

POST /search

Semantic search over resume chunks.

POST /rag

RAG answer using best chunk(s).

POST /chat

Free-form chat using GPT-2.

POST /agent

Routes intelligently between:

Resume search tool

Or normal chat

📂 File Structure
llm-roadmap/
│
├── 07_fastapi_rag_api.py
├── data/sample.pdf
├── chroma_db_resume_api/
├── chroma_db_langchain_resume/
└── README_project2_resume_rag.md

▶️ How to Run

Terminal:

cd ~/Desktop/llm-roadmap
.\.venv\Scripts\activate
uvicorn 07_fastapi_rag_api:app --reload


Open Swagger UI:

http://127.0.0.1:8000/docs

📌 Portfolio Summary

This project shows that Nguakaaga can:

Build embeddings + vector DB

Build RAG systems

Implement LangChain tools & agents

Build a FastAPI backend

Handle offline model loading

Implement full REST API

Structure an LLM engineering project professionally

This is job-ready work.

🎯 Next Step

You are now ready for:

👉 WEEK 8 — Next.js Frontend (Chat UI + File Upload + RAG Interface)
This will complete the resume RAG application end-to-end.