# 03_chroma_search.py
# Mini semantic search engine with SentenceTransformers + ChromaDB

from sentence_transformers import SentenceTransformer
import chromadb
from uuid import uuid4

# Try to support both newer and older Chroma versions
try:
    # Newer API (0.5+)
    from chromadb import PersistentClient
    client = PersistentClient(path="chroma_db")
except ImportError:
    # Older API
    from chromadb.config import Settings
    client = chromadb.Client(
        Settings(
            anonymized_telemetry=False,
            persist_directory="chroma_db",
        )
    )

# 1. Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# 2. Our small "corpus" of texts
documents = [
    "I am learning LLM engineering.",
    "Next.js and Python make a powerful stack.",
    "Vector search finds similar meanings, not exact words.",
    "I want a remote job building AI SaaS products.",
    "Football is my favorite weekend sport.",
]

# 3. Compute embeddings for all documents
print("Encoding documents...")
embeddings = model.encode(documents, convert_to_tensor=True)

# 4. Create / get Chroma collection and add docs
collection = client.get_or_create_collection("study_sentences")

print("Adding documents to Chroma...")
# Use unique IDs every time so we don't need to delete old docs
ids = [f"doc-{i}-{uuid4()}" for i in range(len(documents))]

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),  # tensor -> Python lists
)

print("Docs in collection (might include earlier runs too):", collection.count())
print("Ready for semantic search.\n")


def search_loop():
    """
    Simple interactive loop:
    - Ask user for a query
    - Embed the query
    - Query Chroma
    - Print top matches
    """
    while True:
        query = input("Enter a query (or just press Enter to quit): ").strip()
        if not query:
            print("Goodbye.")
            break

        # Encode query
        query_emb = model.encode(query).tolist()

        # Query Chroma
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
        )

        print(f"\nQuery: {query}\n")
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, dist in zip(docs, dists):
            print(f"Distance: {dist:.4f} | Doc: {doc}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    search_loop()
