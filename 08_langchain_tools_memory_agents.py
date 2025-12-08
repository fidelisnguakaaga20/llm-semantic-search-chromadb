"""
08_langchain_tools_memory_agents.py

Week 6 of the LLM roadmap:
- "Tool"  = a function that searches your resume via Chroma
- "Memory" = a simple conversation history list
- "Agent" = a small class that decides when to call the tool vs just answer

We reuse:
- data/sample.pdf
- chroma_db_langchain_resume/
"""

from pathlib import Path
from typing import List, Dict

from transformers import pipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "sample.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db_langchain_resume"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- Vector DB (same idea as before) ----------

def build_or_load_vectordb():
    """Build the resume vector DB if missing, otherwise load it."""
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    if CHROMA_DIR.exists():
        print("[VECTORDb] Loading existing Chroma DB from:", CHROMA_DIR)
        vectordb = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedder,
        )
    else:
        print("[VECTORDb] Building new Chroma DB from PDF:", PDF_PATH)
        loader = PyPDFLoader(str(PDF_PATH))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        split_docs = splitter.split_documents(docs)

        vectordb = Chroma.from_documents(
            split_docs,
            embedder,
            persist_directory=str(CHROMA_DIR),
        )
        vectordb.persist()

    return vectordb


# ---------- "Tool": resume search ----------

def resume_search(vectordb, query: str, k: int = 3) -> str:
    """
    This is our TOOL.
    It searches your resume in the Chroma vector DB and returns top chunks.
    """
    docs = vectordb.similarity_search(query, k=k)
    if not docs:
        return "No relevant information found in the resume."

    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"--- RESULT {i} ---\n{d.page_content.strip()}")
    return "\n\n".join(parts)


# ---------- Local LLM (gpt2) ----------

def make_local_llm():
    """
    Wrap local gpt2 in a simple callable.

    NOTE: gpt2 is tiny and not instruction-tuned; answers will be messy.
    Goal = wiring tools + memory + agent, not high quality.
    """
    print("[LLM] Loading local gpt2 via HuggingFace pipeline...")
    gen_pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
    )

    def llm_call(prompt: str) -> str:
        out = gen_pipe(prompt)[0]["generated_text"]
        return out

    return llm_call


# ---------- "Agent" with simple memory ----------

class SimpleAgent:
    """
    Very small custom agent:

    - Keeps a conversation history in memory (list of messages).
    - Decides when to call the resume_search TOOL based on keywords.
    - Uses local LLM to answer, optionally including tool results.
    """

    def __init__(self, llm, vectordb):
        self.llm = llm
        self.vectordb = vectordb
        self.history: List[Dict[str, str]] = []  # {"role": "user"/"assistant", "content": "..."}

    def _should_use_resume_tool(self, question: str) -> bool:
        keywords = [
            "resume",
            "cv",
            "my skills",
            "my projects",
            "my experience",
            "my education",
            "according to my resume",
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in keywords)

    def ask(self, question: str) -> str:
        # store user message
        self.history.append({"role": "user", "content": question})

        tool_context = ""
        if self._should_use_resume_tool(question):
            print("[AGENT] Using resume_search TOOL...")
            tool_result = resume_search(self.vectordb, question, k=3)
            tool_context = (
                "The following text was retrieved from the user's resume:\n\n"
                f"{tool_result}\n\n"
            )
        else:
            print("[AGENT] Answering without tools...")

        # build a simple prompt with memory + (optional) tool context
        history_text = ""
        for msg in self.history[-6:]:  # last few turns
            history_text += f"{msg['role'].upper()}: {msg['content']}\n"

        prompt = (
            "You are an assistant helping Nguakaaga think about their career, "
            "LLM engineering roadmap, and resume.\n\n"
            f"{tool_context}"
            "Conversation so far:\n"
            f"{history_text}\n"
            f"ASSISTANT: "
        )

        answer = self.llm(prompt)

        # store assistant answer
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def print_memory(self):
        print("\n[MEMORY CONTENTS]")
        for msg in self.history:
            print(f"{msg['role'].upper()}: {msg['content']}\n")


# ---------- Demos ----------

def demo_agent(vectordb, llm):
    print("\n========== PART 1: SIMPLE AGENT + TOOL ==========\n")
    agent = SimpleAgent(llm=llm, vectordb=vectordb)

    q1 = "What are my strongest skills according to my resume?"
    print(f"\n[USER] {q1}\n")
    a1 = agent.ask(q1)
    print("\n[AGENT ANSWER]\n", a1)

    q2 = "Based on that, what kind of LLM engineering role should I target?"
    print(f"\n[USER] {q2}\n")
    a2 = agent.ask(q2)
    print("\n[AGENT ANSWER]\n", a2)

    agent.print_memory()


def main():
    print("=== Week 6: Tools + Memory + Agent (custom implementation) ===")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    vectordb = build_or_load_vectordb()
    llm = make_local_llm()

    demo_agent(vectordb, llm)

    print("\n=== DONE: Week 6 concepts (tool, memory, agent) are implemented ===")


if __name__ == "__main__":
    main()
