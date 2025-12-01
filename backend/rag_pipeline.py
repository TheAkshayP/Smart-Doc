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
# 🔹 COMPUTE CONFIDENCE SCORE (Simple but Effective)
# ===========================================================
def compute_confidence(docs):
    if not docs:
        return 0.0
    scores = []
    for d in docs:
        s = d.metadata.get("score", 0.5)
        if isinstance(s, (int, float)):
            scores.append(s)
    if not scores:
        return 0.5
    return round(sum(scores) / len(scores), 2)


# ===========================================================
# 🔹 FORMAT FINAL ANSWER BEAUTIFULLY
# ===========================================================
def format_answer(answer, sources, confidence, summary):
    formatted = f"""
📘 **SmartDoc Answer**

{answer}

---

📚 **Sources Used:**
"""
    if sources:
        for i, src in enumerate(sources, 1):
            page = src.metadata.get("page", "Unknown")
            formatted += f"- Source {i}: Page {page}\n"
    else:
        formatted += "- No sources found\n"

    formatted += f"""
🔎 **Confidence:** {confidence}

✨ **TL;DR Summary:**  
{summary}

🧠 _SmartDoc RAG Engine — Powered by Gemini & BGE-Large_
"""
    return formatted.strip()


# ===========================================================
# 🔹 MAIN RAG PIPELINE
# ===========================================================
def answer_question(question: str, vectordb, llm, k=4):
    if vectordb is None:
        return "❌ No document found. Upload a document first."

    if llm is None:
        return "❌ LLM not initialized."

    try:
        # ---------------------------------------------
        # 1) Retrieve relevant chunks
        # ---------------------------------------------
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)

        if not docs:
            return "I don't know"

        context = "\n\n".join(d.page_content for d in docs)
        confidence = compute_confidence(docs)

        # ---------------------------------------------
        # 2) Build improved prompt
        # ---------------------------------------------
        FORMAT_INSTRUCTIONS = """
Format the answer using:
- Bullet points
- Headings
- Short sentences
- **Bold text** for key ideas
- Clean markdown
"""

        prompt = f"""
You are SmartDoc, a strict RAG-based assistant.
Answer ONLY using the context below.
If the answer is NOT in the context, reply: I don't know.

{FORMAT_INSTRUCTIONS}

--------------------------------
📄 CONTEXT:
{context}

❓ QUESTION:
{question}

💡 ANSWER:
"""

        # ---------------------------------------------
        # 3) Generate answer
        # ---------------------------------------------
        print("🔎 Sending prompt to Gemini...")
        response = llm.generate_content(prompt)
        answer = _extract_text_from_genai_response(response).strip()

        # ---------------------------------------------
        # 4) TL;DR Summary
        # ---------------------------------------------
        summary_resp = llm.generate_content(
            f"Summarize this in one short sentence:\n{answer}"
        )
        summary = _extract_text_from_genai_response(summary_resp).strip()

        # ---------------------------------------------
        # 5) Build formatted answer
        # ---------------------------------------------
        return format_answer(answer, docs, confidence, summary)

    except Exception as e:
        return f"❌ RAG Pipeline Error → {e}"
