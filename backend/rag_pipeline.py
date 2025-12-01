import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


# -----------------------------------------------
# Load Gemini LLM (Clean + Correct)
# -----------------------------------------------
def load_llm_pipeline():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY missing in .env")

    model_name = os.getenv("MODEL_NAME", "models/gemini-2.5-flash")

    print(f"🤖 Trying Gemini model: {model_name}")

    genai.configure(api_key=api_key)

    try:
        llm = genai.GenerativeModel(model_name)
        print(f"🤖 Gemini model loaded: {model_name}")
        return llm
    except Exception as e:
        print(f"❌ Failed loading Gemini model: {e}")
        raise e


# -----------------------------------------------
# Extract text safely from Gemini response
# -----------------------------------------------
def _extract_text_from_genai_response(resp):
    try:
        if hasattr(resp, "text"):
            return resp.text

        if hasattr(resp, "candidates") and resp.candidates:
            c = resp.candidates[0]
            if hasattr(c, "content"):
                parts = c.content
                texts = []
                for item in parts:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    elif isinstance(item, str):
                        texts.append(item)
                return "".join(texts)
    except:
        pass

    return str(resp)


# -----------------------------------------------
# RAG Answering (simple & clean)
# -----------------------------------------------
def answer_question(question: str, vectordb, llm, k=4):
    if vectordb is None:
        return "❌ No document found. Upload a document first."

    if llm is None:
        return "❌ LLM not initialized."

    try:
        # Retrieve context from ChromaDB
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)

        context = "\n\n".join(d.page_content for d in docs) if docs else ""

        prompt = f"""
You are SmartDoc, a document analysis expert.
Answer ONLY using the context below.

If the answer is NOT found in the context,
reply exactly: I don't know

--------------------------------
📄 CONTEXT:
{context}

❓ QUESTION:
{question}

💡 ANSWER:
"""

        print("🔎 Sending prompt to Gemini (generate_content)...")
        response = llm.generate_content(prompt)

        answer = _extract_text_from_genai_response(response)
        return answer.strip()

    except Exception as e:
        return f"❌ RAG Pipeline Error → {e}"
