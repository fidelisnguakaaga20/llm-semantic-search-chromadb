# 05_rag_pdf_chroma.py
# Week 5 – First RAG step: PDF -> chunks -> embeddings -> Chroma -> semantic search

from sentence_transformers import SentenceTransformer
import chromadb
from uuid import uuid4
from pypdf import PdfReader
import os


PDF_PATH = "data/sample.pdf"  # <- make sure this file exists


def load_pdf_chunks(pdf_path: str, max_chars: int = 600):
    """
    1) Load PDF
    2) Extract text page by page
    3) Split into small chunks (<= max_chars)
    Returns: list of (chunk_text, metadata_dict)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    reader = PdfReader(pdf_path)
    chunks = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        # simple chunking: break into pieces of ~max_chars
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    (
                        chunk_text,
                        {
                            "page": page_num + 1,
                        },
                    )
                )
            start = end

    return chunks


def get_chroma_client():
    """
    Support both newer and older Chroma APIs (like in 03_chroma_search.py)
    """
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
    return client


def build_pdf_collection(pdf_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load PDF -> chunk -> embed -> store in Chroma collection 'pdf_chunks'
    """
    print(f"Loading PDF from: {pdf_path}")
    chunks = load_pdf_chunks(pdf_path)
    print(f"Total chunks extracted: {len(chunks)}")

    if not chunks:
        raise ValueError("No text chunks were extracted from the PDF.")

    # 1. Load embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # 2. Prepare texts and metadata
    texts = [c[0] for c in chunks]
    metadatas = [c[1] for c in chunks]
    ids = [f"chunk-{i}-{uuid4()}" for i in range(len(chunks))]

    # 3. Embed all chunks
    print("Encoding chunks...")
    embeddings = model.encode(texts, convert_to_tensor=True).tolist()

    # 4. Store in Chroma
    client = get_chroma_client()
    collection = client.get_or_create_collection("pdf_chunks")

    print("Adding chunks to Chroma...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("Chunks in collection (pdf_chunks):", collection.count())
    print("PDF is now indexed for semantic search.\n")

    return collection, model


def search_pdf_loop(collection, model):
    """
    Simple RAG-style retrieval loop:
    - You ask a question
    - We embed it and query Chroma
    - We show the most relevant PDF chunks
    """
    while True:
        query = input("Enter a question about the PDF (or press Enter to quit): ").strip()
        if not query:
            print("Goodbye.")
            break

        query_emb = model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        print(f"\nQuestion: {query}\n")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            page = meta.get("page", "?")
            print(f"Result {i}: (page {page}, distance={dist:.4f})")
            print(doc[:400])
            print("-" * 80)

        # very simple "answer" from top chunk (retrieval-only, no LLM yet)
        if docs:
            print("\n[Naive answer – best PDF chunk]:")
            print(docs[0][:400])
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    collection, model = build_pdf_collection(PDF_PATH)
    search_pdf_loop(collection, model)
