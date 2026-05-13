import os
import gc
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "/tmp/chroma_db"

COLLECTION_NAME = "pdf_collection"
# ----------------------------
# Groq LLM (UPDATED MODEL)
# ----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # ✅ FIXED (was decommissioned)
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# ----------------------------
# Helper → create embeddings fresh each time
# ----------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ----------------------------
# PDF → Vector DB
# ----------------------------
def ingest_pdf(pdf_path):
    gc.collect()

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    docs = splitter.split_documents(pages)
    
    embeddings = get_embeddings()
    
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    try:
        db.delete_collection()
    except:
        pass

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )

    db.persist()
    os.remove(pdf_path)
    return len(docs)

# ----------------------------
# Load Vector DB
# ----------------------------
@lru_cache(maxsize=1)
def load_db():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

# ----------------------------
# Query + Memory + RAG
# ----------------------------
def query(question, history):
    db = load_db()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    # Build conversation history
    recent_history = history[-6:]
    history_text = ""
    for h in recent_history:
        history_text += f"User: {h['user']}\nAssistant: {h['bot']}\n"

    prompt = f"""
You are an AI assistant answering questions from a PDF.

Context:
{context}

Conversation history:
{history_text}

Rules:
- Answer ONLY using the given context
- If answer is not in context, say: "I don't have that information."

Question: {question}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    pages = sorted(set([d.metadata.get("page", 0) + 1 for d in docs]))

    return answer, pages
