# 05_rag_pdf_basic.py
# Month 2 – Week 5: RAG (PDF -> chunks -> embeddings -> Chroma -> PDF-based answer)

import os
from uuid import uuid4

from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
from transformers import pipeline  # LLM imported but not used for now


# ----------------------------
# CONFIG
# ----------------------------

PDF_PATH = "data/sample.pdf"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "pdf_rag_chunks"
LLM_MODEL_NAME = "gpt2"  # kept for later, but not used in answering yet


# ----------------------------
# HELPER: READ + CHUNK PDF
# ----------------------------

def read_pdf(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found at: {path}")

    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    full_text = "\n".join(pages_text)
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Very simple character-based chunking:
    - Take windows of length chunk_size
    - Move forward by (chunk_size - overlap)
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # move with overlap

    # Filter out very small chunks
    chunks = [c for c in chunks if len(c) > 50]
    return chunks


# ----------------------------
# SETUP: EMBEDDINGS + CHROMA
# ----------------------------

print(f"Loading embedding model: {EMBED_MODEL_NAME}")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Setting up Chroma client...")
try:
    from chromadb import PersistentClient
    client = PersistentClient(path="chroma_db")
except ImportError:
    from chromadb.config import Settings
    client = chromadb.Client(
        Settings(
            anonymized_telemetry=False,
            persist_directory="chroma_db",
        )
    )

collection = client.get_or_create_collection(CHROMA_COLLECTION)

# (We keep this for later when we plug in a better chat model)
print(f"Loading LLM placeholder (not used yet): {LLM_MODEL_NAME}")
_ = pipeline("text-generation", model=LLM_MODEL_NAME)


# ----------------------------
# STEP 1: LOAD + CHUNK PDF
# ----------------------------

print(f"\nReading PDF from: {PDF_PATH}")
pdf_text = read_pdf(PDF_PATH)
print(f"Total characters in PDF: {len(pdf_text)}")

print("Chunking text...")
chunks = chunk_text(pdf_text, chunk_size=500, overlap=100)
print(f"Total chunks created: {len(chunks)}")

if len(chunks) == 0:
    raise ValueError("No chunks were created from the PDF. Check if the PDF has selectable text.")


# ----------------------------
# STEP 2: EMBED + STORE IN CHROMA
# ----------------------------

print("\nEncoding chunks into embeddings...")
embeddings = embed_model.encode(chunks, convert_to_tensor=True)

print("Adding chunks to Chroma collection (with unique IDs)...")
ids = [f"chunk-{i}-{uuid4()}" for i in range(len(chunks))]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist(),
)

print("Chunks in collection (might include earlier runs too):", collection.count())
print("RAG index is ready.\n")


# ----------------------------
# STEP 3: QA LOOP – RETRIEVAL + DIRECT PDF ANSWER
# ----------------------------

def answer_question(question: str, k: int = 3):
    # 1) Embed the question
    q_emb = embed_model.encode(question).tolist()

    # 2) Retrieve top-k chunks from Chroma
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        print("No chunks retrieved. Try another question.")
        return

    print("\nTop retrieved chunks:")
    for i, (doc, dist) in enumerate(zip(docs, distances), start=1):
        print(f"\n--- Chunk {i} (distance={dist:.4f}) ---")
        print(doc[:400], "...\n")  # print first 400 chars

    # 3) Use the BEST chunk as the answer (no hallucinations)
    best_chunk = docs[0]

    print("=== ANSWER FROM PDF (BEST CHUNK) ===\n")
    print(best_chunk)
    print("\n====================\n")


def main():
    print("PDF RAG ready. Ask questions about the PDF.\n")
    while True:
        q = input("Enter a question (or blank to quit): ").strip()
        if not q:
            print("Goodbye.")
            break
        answer_question(q, k=3)


if __name__ == "__main__":
    main()
