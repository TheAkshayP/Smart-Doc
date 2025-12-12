import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ===========================================================
# 🔹 LOAD GEMINI LLM
# ===========================================================
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


# ===========================================================
# 🔹 SAFE EXTRACT TEXT FROM GEMINI RESPONSE
# ===========================================================
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


# ===========================================================
# 🔹 FORMAT ANSWER (Handles both known + unknown questions)
# ===========================================================
def format_answer(answer, sources):

    # Normalize lowercase for detection
    ans_low = answer.lower().strip()

    # ---------------------------------------------
    # CASE 1: If the answer is "I don't know" (any variation)
    # ---------------------------------------------
    if ("i don't know" in ans_low) or ("i dont know" in ans_low) or ("i do not know" in ans_low):
        formatted = f"""
I don't know.

This question does not seem to be related to the content of the uploaded PDF.

🧠 SmartDoc RAG Engine — Made by Akshay
"""
        return formatted.strip()

    # ---------------------------------------------
    # CASE 2: Normal PDF-related answer
    # ---------------------------------------------
    if sources:
        pages = []
        for src in sources:
            page = src.metadata.get("page", "Unknown")
            pages.append(f"- Page {page}")
        source_text = "\n".join(pages)
    else:
        source_text = "- No sources found"

    formatted = f"""

{answer}

📚 **Sources Used:**  
{source_text}

🧠 SmartDoc RAG Engine — Made by Akshay
"""
    return formatted.strip()


# ===========================================================
# 🔹 MAIN RAG PIPELINE
# ===========================================================
def answer_question(question: str, vectordb, llm, k=4):

    # ----------------------------
    # Handle missing vector DB
    # ----------------------------
    if vectordb is None:
        return format_answer("I don't know", [])

    # ----------------------------
    # Handle missing LLM
    # ----------------------------
    if llm is None:
        return format_answer("I don't know", [])

    try:
        # ---------------------------------------------
        # 1. Retrieve relevant chunks
        # ---------------------------------------------
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)

        # If RAG finds nothing → unrelated question
        if not docs:
            return format_answer("I don't know", [])

        # ---------------------------------------------
        # 2. Build context
        # ---------------------------------------------
        context = "\n\n".join(d.page_content for d in docs)

        # ---------------------------------------------
        # 3. Build prompt
        # ---------------------------------------------
        prompt = f"""
You are SmartDoc, a RAG-based assistant.
Answer ONLY using the context below.
If the answer is NOT in the context, reply exactly: I don't know.

Context:
{context}

Question:
{question}

Answer:
"""

        # ---------------------------------------------
        # 4. Generate Answer
        # ---------------------------------------------
        print("🔎 Sending prompt to Gemini...")
        response = llm.generate_content(prompt)
        answer = _extract_text_from_genai_response(response).strip()

        # ---------------------------------------------
        # 5. Format final answer (normal or unknown)
        # ---------------------------------------------
        return format_answer(answer, docs)

    except Exception as e:
        return f"❌ RAG Pipeline Error → {e}"
