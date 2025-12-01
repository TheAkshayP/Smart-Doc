import sys
import os
import time
import shutil
from dotenv import load_dotenv

# Disable telemetry BEFORE any google import
import disable_telemetry  # noqa

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local utilities
from utils.document_loader import load_document
from utils.vector_store import (
    store_embeddings,
    load_existing_embeddings,
    split_into_chunks,
    close_vectordb,   # <-- IMPORTANT
)
from rag_pipeline import load_llm_pipeline, answer_question


# =====================================================
# Load environment variables
# =====================================================
load_dotenv()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vectorstore")
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vectordb = None
llm = None


# =====================================================
# FastAPI Application
# =====================================================
app = FastAPI(title="SmartDoc Backend", version="3.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


# =====================================================
# UPLOAD ENDPOINT
# =====================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectordb

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"📄 Uploaded: {file.filename}")
    except Exception as e:
        return {"message": f"❌ Error saving file: {e}"}

    # Extract text
    try:
        full_text = load_document(file_path)
    except Exception as e:
        return {"message": f"❌ Failed to extract text: {e}"}

    if not full_text.strip():
        return {"message": "❌ No readable text in document."}

    # Chunk text
    chunks = split_into_chunks(full_text)

    # Store embeddings
    try:
        vectordb = store_embeddings(chunks, persist_dir=VECTOR_DB_PATH)
    except Exception as e:
        return {"message": f"❌ Failed storing embeddings: {e}"}

    return {"message": "File uploaded & processed successfully!"}


# =====================================================
# ASK ENDPOINT
# =====================================================
@app.post("/ask")
async def ask(query: Query):
    global vectordb, llm

    # Ensure vector DB exists
    if vectordb is None:
        vectordb = load_existing_embeddings(VECTOR_DB_PATH)
        if vectordb is None:
            return {"answer": "❌ Please upload a document first."}

    # Load LLM if not loaded
    if llm is None:
        print("⏳ Loading Gemini LLM...")
        llm = load_llm_pipeline()
        if llm is None:
            return {"answer": "❌ Failed to initialize LLM."}
        print("🤖 Gemini LLM Ready!")

    try:
        # Determine how many chunks exist
        try:
            available = vectordb._collection.count()
        except:
            available = 1

        default_k = int(os.getenv("TOP_K", 5))
        safe_k = min(default_k, max(1, available))

        print(f"🔍 Chroma search → k={safe_k}/{available}")

        answer = answer_question(
            question=query.question,
            vectordb=vectordb,
            llm=llm,
            k=safe_k,
        )

        return {"answer": answer}

    except Exception as e:
        print(f"❌ Error in /ask: {e}")
        return {"answer": "Something went wrong while generating the answer."}


# =====================================================
# RESET ENDPOINT (Windows-safe)
# =====================================================
@app.post("/reset")
async def reset():
    global vectordb, llm

    # 1️⃣ Release Chroma file locks
    close_vectordb(vectordb)

    vectordb = None
    llm = None

    # 2️⃣ Delete uploaded docs
    try:
        for f in os.listdir(UPLOAD_DIR):
            try:
                os.remove(os.path.join(UPLOAD_DIR, f))
            except:
                pass
    except:
        pass

    # 3️⃣ Delete vectorstore directory
    if os.path.exists(VECTOR_DB_PATH):
        try:
            shutil.rmtree(VECTOR_DB_PATH)
            print("🗑 Vector DB deleted successfully.")
        except Exception as e:
            print(f"❌ Delete failed: {e} — retrying in 1s...")
            time.sleep(1)
            try:
                shutil.rmtree(VECTOR_DB_PATH)
                print("🗑 Vector DB deleted on retry.")
            except Exception as e2:
                print(f"❌ Retry failed: {e2}")
                return {"message": f"Failed to reset vector DB: {e2}"}

    return {"message": "SmartDoc reset successfully!"}


# =====================================================
# ROOT ROUTE
# =====================================================
@app.get("/")
def home():
    return {"message": "SmartDoc backend running (Gemini RAG v3.4)!"}
