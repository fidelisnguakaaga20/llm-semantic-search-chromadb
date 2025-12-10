# 06_langchain_resume_rag.py
# Month 2 – Week 6: LangChain RAG over your resume (no langchain.chains)

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ----------------------------
# CONFIG
# ----------------------------

PDF_PATH = "data/sample.pdf"  # your resume PDF

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# /// IMPORTANT: this must match the FastAPI backend
CHROMA_DIR = "chroma_db_resume_api"          # was: "chroma_db_langchain_resume"
CHROMA_COLLECTION_NAME = "resume_chunks"     # explicit collection name

LLM_MODEL_NAME = "gpt2"  # small local model, just to demonstrate the pipeline


# ----------------------------
# 1) LOAD PDF WITH LANGCHAIN
# ----------------------------

def load_resume_docs(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume PDF not found at: {pdf_path}")

    print(f"Loading PDF with LangChain PyPDFLoader: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from PDF.")
    return docs


# ----------------------------
# 2) SPLIT INTO CHUNKS
# ----------------------------

def split_docs(docs):
    print("Splitting documents with RecursiveCharacterTextSplitter...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    split = splitter.split_documents(docs)
    print(f"Total chunks created: {len(split)}")
    return split


# ----------------------------
# 3) BUILD VECTORSTORE (CHROMA)
# ----------------------------

def build_vectorstore(chunks):
    print(f"Creating HuggingFace embeddings: {EMBED_MODEL_NAME}")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    print(f"Building Chroma vectorstore at: {CHROMA_DIR}")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_DIR,          # /// same dir as FastAPI
        collection_name=CHROMA_COLLECTION_NAME # /// same collection as FastAPI
    )

    # For newer Chroma, persistence is automatic, but this is harmless.
    try:
        vectordb.persist()
    except Exception:
        pass

    try:
        count = vectordb._collection.count()
    except Exception:
        count = "unknown"

    print(f"Vectorstore ready. Total vectors: {count}")
    return vectordb, embedder


# ----------------------------
# 4) LOCAL LLM VIA HUGGINGFACE
# ----------------------------

def load_local_llm():
    print(f"Loading local LLM: {LLM_MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=120,
    )
    return gen_pipe


# ----------------------------
# 5) RAG ANSWERING FUNCTION
# ----------------------------

def answer_question(question: str, vectordb: Chroma, llm_pipe):
    # 1) Create retriever from vectorstore
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # ✅ New LangChain style: retriever is a Runnable → use .invoke()
    docs = retriever.invoke(question)

    print("\nTop retrieved chunks:\n")
    for i, d in enumerate(docs, start=1):
        print(f"--- Chunk {i} ---")
        print(d.page_content[:400], "...\n")

    # 2) Build prompt from context
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "You are answering questions about my resume.\n"
        "Use ONLY the context below. If the answer is not there, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    print("Generating answer from local LLM...\n")
    out = llm_pipe(prompt)[0]["generated_text"]

    # Extract text after "Answer:"
    if "Answer:" in out:
        answer_part = out.split("Answer:", 1)[1].strip()
    else:
        answer_part = out.strip()

    print("=== FINAL ANSWER (gpt2 is small, so expect messy text) ===\n")
    print(answer_part)
    print("\n=========================================\n")


# ----------------------------
# MAIN CLI LOOP
# ----------------------------

def main():
    # 1) Load + split resume
    docs = load_resume_docs(PDF_PATH)
    chunks = split_docs(docs)

    # 2) Build / load vectorstore
    vectordb, _ = build_vectorstore(chunks)

    # 3) Local LLM
    llm_pipe = load_local_llm()

    # 4) Interactive questions
    print("\n[LangChain RAG] Ask questions about your resume.")
    print("Press Enter on empty line to quit.\n")

    while True:
        q = input("Enter a question: ").strip()
        if not q:
            print("Goodbye.")
            break
        answer_question(q, vectordb, llm_pipe)


if __name__ == "__main__":
    main()
