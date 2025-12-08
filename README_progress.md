# ✅ PROGRESS SO FAR (LLM ROADMAP)

🔥 MONTH 1 — Python + Transformers + Embeddings

**WEEK 1: Python – DONE**

- Basic syntax: variables, lists, dicts
- Functions and modules
- File handling (`02_files.py` writing/reading `study_plan.txt`)
- Jupyter basics in `01_python_basics.ipynb`

**WEEK 2: HuggingFace Basics – DONE**

- Loaded GPT-2 with `AutoTokenizer` + `AutoModelForCausalLM`
- Tokenized text and inspected `input_ids` + tokens
- Generated text with `pipeline("text-generation", model="gpt2")`
- Extracted embeddings by:
  - taking `outputs.hidden_states[-1]`
  - mean-pooling over sequence to get sentence embeddings

**WEEK 3: Embeddings + Vector Search – DONE**

- Sentence Transformers model: `sentence-transformers/all-MiniLM-L6-v2`
- Local Chroma DB (`chroma_db/`) with collection for study sentences
- Semantic search script: `03_chroma_search.py`
  - CLI: type a query → embed → Chroma → see most similar sentences
- Notebook version: `03_embeddings_search.ipynb`
- Pushed as **Project 1 – Semantic Search with SentenceTransformers + ChromaDB**
  - Repo: `llm-semantic-search-chromadb`

**WEEK 4: Transformer Concepts – DONE**

- Script: `04_transformer_concepts.py`
- Demonstrated:
  - Tokenization (subwords, IDs)
  - Attention shapes and weights (last layer, specific head)
  - High-level architecture of GPT-style decoder blocks (embeddings, masked self-attention, MLP, residuals, LayerNorm)

---

🔥 MONTH 2 — RAG + LangChain + Backend

**WEEK 5: RAG – PDF → Chunks → Embeddings → Chroma – DONE**

- Resume PDF stored at `data/sample.pdf`
- Script: `data/05_rag_pdf_basic.py`
- Steps:
  - Load PDF with `pypdf.PdfReader`
  - Chunk text into overlapping windows
  - Embed chunks with `SentenceTransformer("all-MiniLM-L6-v2")`
  - Store in Chroma collection (`pdf_rag_chunks`)
  - CLI Q&A: type a question → retrieve top-k chunks → return best chunk as answer

**WEEK 6: LangChain RAG – DONE (Local CLI)**

- Script: `06_langchain_resume_rag.py`
- Uses **LangChain**:
  - `PyPDFLoader` to load `data/sample.pdf`
  - `RecursiveCharacterTextSplitter` for chunking
  - `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)
  - `Chroma` vector store (`chroma_db_langchain_resume`)
- RAG loop:
  - You type a question in the terminal
  - It retrieves the most relevant chunks
  - Uses a **local GPT-2** pipeline to generate a noisy answer
  - Shows both the retrieved context and final answer

**WEEK 7: FastAPI Backend – IN PROGRESS (Core Endpoints DONE)**

- Script: `07_fastapi_resume_api.py`
- Tech:
  - FastAPI + Uvicorn
  - Chroma (same resume chunks)
  - Sentence Transformers (`all-MiniLM-L6-v2`)
- Implemented endpoints:
  - `GET /health` – health check
  - `POST /embed` – embed arbitrary text
  - `POST /search` – semantic search over resume chunks
  - `POST /rag` – simple RAG:
    - retrieve top-k chunks
    - return best chunk as `answer` plus context list
- Tested via FastAPI docs at `http://127.0.0.1:8000/docs`:
  - `/search` → returns top chunks + distances
  - `/rag` → returns `answer` (best chunk) + `context`
- This is **Project 2 – Resume RAG API (LangChain + FastAPI + Chroma)**

Next: finish Week 7 (optional `/chat`, `/agent` stubs) and move to **Week 8 – Next.js frontend integration**.


<!-- # LLM Roadmap – Progress Log

This file tracks my progress against the MASTER LLM ENGINEERING ROADMAP.

- Month 1 – Week 1: Python ✅
- Month 1 – Week 2: HuggingFace Basics ✅
- Month 1 – Week 3: Embeddings + Vector Search ✅
- Month 1 – Week 4: Transformer Concepts ✅
- Month 2 – Week 5: RAG (PDF → chunks → embeddings → Chroma) ✅
- Month 2 – Week 6: LangChain RAG over my resume ✅
- Month 2 – Week 7: FastAPI Resume RAG API 🚧 (in progress)


# ✅ PROGRESS SO FAR (UP TO MONTH 2, WEEK 6)

## Month 1 — Python + Transformers + Embeddings

### WEEK 1: Python ✅
Covered exactly as planned:

- **Variables / types / printing**
  - `01_basics.py`
- **Lists & dicts**
  - Create, update, index, loop over them
- **Functions**
  - Simple functions + parameters, return values
- **File handling**
  - `02_files.py` → writes & reads `study_plan.txt`
- **Jupyter Notebook**
  - Opened notebook, ran cells, printed outputs

> Status: ✅ Week 1 fully complete.

---

### WEEK 2: HuggingFace Basics ✅
All 4 bullets done using **GPT-2** locally:

- **Load model**
  - `test_llm.py` + `02_hf_basics.ipynb`  
  - `AutoTokenizer`, `AutoModelForCausalLM`, and `pipeline("text-generation", model="gpt2")`
- **Tokenize text**
  - Inspected `input_ids`, tokens, and decoded text in notebook
- **Generate text**
  - Generated continuations for prompts like  
    `"Learning LLMs with confidence:"`
- **Extract embeddings**
  - Took `hidden_states[-1]` from the model and mean-pooled to get a sentence embedding

> Status: ✅ Week 2 fully complete.

---

### WEEK 3: Embeddings + Vector Search ✅
Used **SentenceTransformers + ChromaDB** and built a real mini-project.

- **Use Sentence Transformers**
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Store embeddings in Chroma**
  - Created local `chroma_db` and a collection
- **Query by similarity**
  - `collection.query(...)` with `query_embeddings`
- **Build simple search engine**
  - `03_chroma_search.py` → interactive CLI:
    - Enter query → encode → search in Chroma → show most similar sentences
  - `03_embeddings_search.ipynb` → same logic in notebook
- **Project saved to GitHub**
  - Project: **Semantic Search with SentenceTransformers + Chroma**
  - Repo: `llm-semantic-search-chromadb`

> Status: ✅ Week 3 fully complete and already portfolio-ready (Project 1).

---

### WEEK 4: Transformers Concepts ✅
Focused on **understanding GPT-style transformers** in code.

- **Tokenization (deeper)**
  - `04_transformer_concepts.py`
  - Showed how a sentence turns into sub-tokens + IDs
- **Attention (basic)**
  - Enabled model attentions and printed:
    - attention shape: `(batch, heads, seq_len, seq_len)`
    - which tokens the last token attends to most
- **Decoder models (GPT-style)**
  - Explained:
    - input embeddings + positional embeddings
    - masked self-attention (no look-ahead)
    - feed-forward layers
    - residuals + layer norm
    - stacked decoder blocks → logits for next token

> Status: ✅ Week 4 fully complete (good mental model of how GPT-like models work).

---

## Month 2 — RAG + LangChain + Backend

### WEEK 5: RAG (Manual PDF → Chunks → Embeddings → Chroma → Answer) ✅
Built a **manual RAG pipeline** over your own resume PDF.

- **PDF loader**
  - `05_rag_pdf_basic.py` using `pypdf.PdfReader`
  - Reads `data/sample.pdf` (your resume)
- **Chunking**
  - Simple character-based chunks with overlap
- **Embedding chunks**
  - SentenceTransformers: `all-MiniLM-L6-v2`
- **Vector DB indexing**
  - Stores chunk embeddings into **Chroma** collection `pdf_rag_chunks`
- **Retrieval + answer**
  - For each question:
    - Embed question → query Chroma → get top chunks
    - Print best chunk as **“ANSWER FROM PDF (BEST CHUNK)”**

Examples you ran:

- “What are my top projects?”
- “What tech stack do I use?”
- “What is my phone number?”

> Status: ✅ Week 5 complete.  
> You now have a working **resume RAG system** without LangChain.

---

### WEEK 6: LangChain / RAG Pipeline (FIRST PART DONE) ✅
Rebuilt the resume RAG pipeline using **LangChain** on top of the same ideas.

- **Chains / RAG pipeline (core retrieval + LLM)** ✅
  - File: `06_langchain_resume_rag.py`
  - Uses:
    - `PyPDFLoader` to load `data/sample.pdf`
    - `RecursiveCharacterTextSplitter` for chunking
    - `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)
    - `Chroma` vectorstore (`chroma_db_langchain_resume`)
    - `.as_retriever()` to get relevant chunks
  - For each CLI question:
    - Prints **Top retrieved chunks**
    - Uses local `gpt2` via `pipeline("text-generation")` to generate an answer
    - Note: `gpt2` is tiny, so answers are messy — this is expected for now.  
      The important part is **LangChain + vectorstore + retrieval** are working.

Questions you tested:

- “What are my top projects?”
- “What tech stack do I use?”
- “Where can recruiters see my live projects?”

> Status: ✅ RAG pipeline with LangChain is working over your resume.  
> Remaining Week 6 topics (tools, memory, more advanced agents) will be layered on top later.

---

## Current Portfolio Projects From This Roadmap

1. **Project 1 – Semantic Search with SentenceTransformers + Chroma (CLI)**
   - Stack: Python, `sentence-transformers`, `chromadb`
   - Features: encodes sentences, stores in Chroma, CLI semantic search.
   - Repo: `llm-semantic-search-chromadb`

2. **Project 2 – Resume Q&A RAG (Manual + LangChain, CLI)**
   - Stack: Python, `pypdf`, `sentence-transformers`, `chromadb`, `langchain-community`, `langchain`, local `gpt2`
   - Features:
     - Load your PDF resume
     - Chunk → embed → store in Chroma
     - Ask questions; retrieve best chunks about your skills/projects/stack
     - LangChain version wraps the same logic into a retriever + simple chain

You are now **fully on track up to:**

- ✅ Month 1: Weeks 1–4  
- ✅ Month 2: Week 5 and the **core RAG pipeline part of Week 6**

Next steps (still following the roadmap):

- Finish the rest of **Week 6** (tools, memory, simple agents on top of this RAG).
- Then move into **Week 7: FastAPI backend** (`/embed`, `/search`, `/rag`, `/chat`, `/agent`). -->
