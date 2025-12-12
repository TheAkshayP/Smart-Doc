import json
import requests
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_similarity,
    faithfulness,
)

# ---------- LOCAL EMBEDDINGS ----------
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- LOCAL JUDGE LLM ----------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


# ==========================================================
# CONFIG
# ==========================================================
BACKEND = "http://127.0.0.1:8000"
EVAL_FILE = "eval/data.json"
K = 5


# ==========================================================
# BACKEND CALLS
# ==========================================================
def call_ask(question):
    resp = requests.post(f"{BACKEND}/ask", json={"question": question}, timeout=60)
    if resp.ok:
        return resp.json().get("answer", "")
    return ""


def call_retrieve(question, k=K):
    resp = requests.post(f"{BACKEND}/retrieve", json={"question": question, "k": k}, timeout=30)
    if resp.ok:
        return resp.json().get("contexts", [])
    return []


# ==========================================================
# BUILD DATASET
# ==========================================================
def build_rows(data):
    rows = []
    for item in data:
        q = item["question"]
        ground_truth = item.get("ground_truth", "")

        answer = call_ask(q)
        contexts = call_retrieve(q, K)

        formatted_contexts = [
            {"text": c.get("content", ""), "meta": c.get("metadata", {})}
            for c in contexts
        ]

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": formatted_contexts,
            "ground_truth": ground_truth
        })

    return rows


# ==========================================================
# MAIN
# ==========================================================
def main():
    data = json.load(open(EVAL_FILE, "r", encoding="utf-8"))
    rows = build_rows(data)
    ds = Dataset.from_list(rows)

    print("\n🔥 Running SmartDoc RAG Evaluation (NO OpenAI)...\n")

    # ----------------- Embeddings -----------------
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ----------------- Judge LLM -----------------
    print("Loading local evaluation model: flan-t5-base ...")

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128
    )

    judge_llm = HuggingFacePipeline(pipeline=pipe)

    # ----------------- Run RAGAS -----------------
    results = evaluate(
        ds,
        embeddings=embed_model,
        llm=judge_llm,
        metrics=[
            context_precision,
            context_recall,
            answer_similarity,
            faithfulness,
        ]
    )

    print("\n--- FINAL RAGAS RESULTS ---")
    print(results.to_pandas().to_json(indent=2))


if __name__ == "__main__":
    main()
