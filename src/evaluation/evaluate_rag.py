"""
--------------------------------------------------------------
 Evaluate Legal RAG Model (Retrieval + Generation)
--------------------------------------------------------------
Author: Raktim
Purpose:
    - Evaluate retrieval and generation performance of the RAG system
    - Compute: Precision@K, cosine similarity, hallucination rate
    - Automatically version and log results
--------------------------------------------------------------
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------
# CLI Arguments
# --------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate and version RAG model performance.")
parser.add_argument("--note", type=str, default="", help="Optional description for this version (e.g. changed retriever k=10)")
args = parser.parse_args()

# --------------------------------------------------------------
# Setup
# --------------------------------------------------------------
DB_DIR = "data/vector_db/"
EVAL_FILE = "data/processed/legal_eval_sample.json"
LOG_FILE = "evaluation/version_log.json"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

LLM_NAME = "HuggingFaceH4/zephyr-7b-beta"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(model=LLM_NAME, token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
eval_model = SentenceTransformer(EMBED_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

prompt_template = ChatPromptTemplate.from_template("""
You are a helpful Indian legal assistant.
Answer the user's question based **only** on the provided legal context.
If the context lacks information, clearly say:
"The provided context does not have enough information to answer precisely."

Context:
{context}

Question:
{question}

Answer:
""")

# --------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------
def cosine_sim(a, b):
    return util.cos_sim(a, b).item()

def extract_answer(text):
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()

def get_next_version(log_file):
    """Auto-generate next version ID (v1.0, v1.1, etc.)"""
    if not os.path.exists(log_file):
        return "v1.0"
    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not data:
            return "v1.0"
        last_ver = data[-1]["version"]
        prefix, num = last_ver[0], float(last_ver[1:])
        return f"v{num + 0.1:.1f}"

def evaluate_sample(sample, k=5):
    """Evaluate one sample for retrieval and generation."""
    question, gold_answer = sample["question"], sample["answer"]

    # Retrieval
    results = db.similarity_search_with_score(question, k=k)
    retrieved_texts = [extract_answer(doc.page_content) for doc, _ in results]
    retrieved_embs = eval_model.encode(retrieved_texts)
    q_emb = eval_model.encode(question)
    sims = [util.cos_sim(q_emb, e).item() for e in retrieved_embs]

    retrieval_precision = 1 if any(
        util.cos_sim(eval_model.encode(gold_answer), e).item() > 0.6
        for e in retrieved_embs
    ) else 0

    # Generation
    context_text = "\n\n".join(retrieved_texts[:3])
    prompt = prompt_template.format(context=context_text, question=question)

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful Indian legal assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.3
    )
    model_answer = response.choices[0].message["content"]

    gold_emb = eval_model.encode(gold_answer)
    model_emb = eval_model.encode(model_answer)
    answer_similarity = cosine_sim(gold_emb, model_emb)

    hallucinated = int(
        "not have enough information" in model_answer.lower()
        or "i do not know" in model_answer.lower()
    )

    return {
        "retrieval_precision": retrieval_precision,
        "answer_similarity": answer_similarity,
        "hallucinated": hallucinated
    }

# --------------------------------------------------------------
# Evaluation Loop
# --------------------------------------------------------------
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"📘 Evaluating {len(eval_data)} samples...\n")

results = []
for sample in tqdm(eval_data):
    try:
        res = evaluate_sample(sample)
        results.append(res)
    except Exception as e:
        print(f"⚠️ Error: {e}")

# --------------------------------------------------------------
# Aggregate Metrics
# --------------------------------------------------------------
retrieval_precision = np.mean([r["retrieval_precision"] for r in results])
avg_similarity = np.mean([r["answer_similarity"] for r in results])
hallucination_rate = np.mean([r["hallucinated"] for r in results])

print("\n📊 Evaluation Summary:")
print(f"✅ Retrieval Precision@5: {retrieval_precision:.3f}")
print(f"✅ Avg. Answer Similarity: {avg_similarity:.3f}")
print(f"⚠️ Hallucination Rate: {hallucination_rate:.3f}")

# --------------------------------------------------------------
# Version Logging
# --------------------------------------------------------------
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

db_version = datetime.fromtimestamp(os.path.getmtime(DB_DIR)).strftime("%Y-%m-%d %H:%M:%S")
version_id = get_next_version(LOG_FILE)

entry = {
    "version": version_id,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "llm_model": LLM_NAME,
    "embedding_model": EMBED_MODEL,
    "db_version": db_version,
    "retrieval_precision": round(retrieval_precision, 3),
    "answer_similarity": round(avg_similarity, 3),
    "hallucination_rate": round(hallucination_rate, 3),
    "num_samples": len(eval_data),
    "note": args.note or ""
}

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log_data = json.load(f)
else:
    log_data = []

log_data.append(entry)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    json.dump(log_data, f, indent=4)

print(f"\n💾 Logged as version {version_id} to {LOG_FILE}")
if args.note:
    print(f"📝 Note: {args.note}")
