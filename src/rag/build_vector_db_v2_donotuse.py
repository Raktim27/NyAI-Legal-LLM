import os
import json
import shutil
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DATA_PATH = "data/processed/legal_qa_clean.json"
DB_DIR = "data/vector_db/"

# Rebuild DB fresh
if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)
os.makedirs(DB_DIR, exist_ok=True)

# Load JSON
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

clean_docs = []
seen_answers = set()

print("Cleaning and preparing documents...")

for item in tqdm(data):
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()

    if not answer:
        continue

    answer_norm = " ".join(answer.split())

    # Prevent duplicate answers
    if answer_norm.lower() in seen_answers:
        continue

    seen_answers.add(answer_norm.lower())

    # Store ONLY answer, NOT question
    clean_docs.append(Document(page_content=answer_norm))

print(f"Prepared {len(clean_docs)} clean legal documents.")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building vector store...")
vectorstore = Chroma.from_documents(
    documents=clean_docs,
    embedding=embedding_model,
    persist_directory=DB_DIR
)

vectorstore.persist()
print("Vector database saved successfully!")
