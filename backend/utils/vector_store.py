import os
import shutil
import gc
import time
import stat
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# ============================================================
# 🔹 LOAD EMBEDDING MODEL  (UPDATED FOR BGE-LARGE)
# ============================================================
def get_embedding_model():
    """
    Loads the embedding model defined in .env.
    Defaults to BGE-Large if nothing is set.
    """
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    print(f"🔧 Loading Embedding Model: {model_name}")

    return SentenceTransformerEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True}  # Required for BGE!
    )


# ============================================================
# 🔹 CHUNKING (NO CHANGE)
# ============================================================
def split_into_chunks(full_text: str, chunk_size: int = 500, overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(full_text)
    if not chunks:
        chunks = [full_text]

    print(f"📝 Total Chunks Created: {len(chunks)}")
    return chunks


# ============================================================
# 🔹 SAFE CLOSE OF CHROMA
# ============================================================
def safe_close_vectordb(vectordb):
    if vectordb is None:
        return

    print("🔒 [SAFE CLOSE] Closing Chroma DB...")

    try:
        if hasattr(vectordb, "_client"):
            try:
                vectordb._client._persist_client = False
            except:
                pass
            vectordb._client = None

        if hasattr(vectordb, "_collection"):
            vectordb._collection = None

    except Exception as e:
        print(f"⚠️ [SAFE CLOSE] Failed closing internals: {e}")

    try:
        del vectordb
    except:
        pass

    gc.collect()
    time.sleep(0.4)
    print("🔓 [SAFE CLOSE] Released.")


# ============================================================
# 🔹 AGGRESSIVE DIRECTORY REMOVAL
# ============================================================
def force_remove_dir(path: str, retries=6):
    if not os.path.exists(path):
        return True

    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path)
            return True
        except Exception as e:
            print(f"❌ Delete failed (attempt {attempt}/{retries}): {e}")

            # Remove file attributes + retry delete
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    fpath = os.path.join(root, name)
                    try:
                        os.chmod(fpath, stat.S_IWRITE)
                        os.remove(fpath)
                    except:
                        pass

                for name in dirs:
                    dpath = os.path.join(root, name)
                    try:
                        os.rmdir(dpath)
                    except:
                        pass

            gc.collect()
            time.sleep(0.2 * attempt)

    try:
        shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"❌ Final delete failure: {e}")
        return False


# ============================================================
# 🔹 HYBRID CLOSE → SAFE THEN FORCE REMOVE DIRECTORY
# ============================================================
def close_vectordb(vectordb, persist_dir=None):
    safe_close_vectordb(vectordb)

    if persist_dir and os.path.exists(persist_dir):
        if force_remove_dir(persist_dir):
            print("🗑 Chroma directory removed successfully.")
        else:
            print("❌ Could not delete Chroma directory even after aggressive removal.")


# ============================================================
# 🔹 LOAD EXISTING CHROMA DB
# ============================================================
def load_existing_embeddings(persist_dir: str):
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print("⚠️ No existing vector database found.")
        return None

    try:
        print(f"📂 Loading vector DB → {persist_dir}")

        embedding_model = get_embedding_model()
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )

        print("✅ Vector DB loaded successfully.")
        return vectordb

    except Exception as e:
        print(f"❌ Error loading vector DB: {e}")
        return None


# ============================================================
# 🔹 CREATE / UPDATE CHROMA VECTORSTORE (UPDATED FOR BGE)
# ============================================================
def store_embeddings(chunks, persist_dir: str):
    if not chunks:
        raise ValueError("❌ No text chunks provided.")

    os.makedirs(persist_dir, exist_ok=True)
    embedding_model = get_embedding_model()

    # UPDATE EXISTING DB
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("🔍 Existing DB found → updating...")

        try:
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_model
            )

            vectordb.add_texts(chunks)

            try:
                vectordb.persist()
            except:
                pass

            print(f"📌 Added {len(chunks)} new chunks.")
            print("💾 Vector DB updated successfully.")
            return vectordb

        except Exception as e:
            print(f"⚠️ Error updating DB: {e}")

            # RESET DB ON DIMENSION MISMATCH
            if "dimension" in str(e).lower():
                print("⚠️ Dimension mismatch → resetting DB...")
                close_vectordb(vectordb, persist_dir)

            else:
                raise

    # CREATE NEW DB
    print("📁 Creating NEW vector DB...")

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

    try:
        vectordb.persist()
    except:
        pass

    print("✅ New DB created!")
    return vectordb
