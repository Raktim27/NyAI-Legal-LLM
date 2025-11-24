"""
--------------------------------------------------------------
 Build Vector Database for Legal Q&A (RAG Component)
--------------------------------------------------------------
Author: Raktim
Purpose:
    - Convert cleaned Q&A dataset into dense embeddings
    - Store them in a local Chroma vector database
    - Used for Retrieval-Augmented Generation (RAG)
--------------------------------------------------------------
Expected Inputs:
    data/processed/legal_qa_clean.json
Outputs:
    vector_db/ (Chroma directory with embeddings)
--------------------------------------------------------------
"""

import os
import json
import shutil
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --------------------------------------------------------------
# Step 1. Configuration
# --------------------------------------------------------------
DATA_PATH = "data/processed/legal_qa_clean.json"
DB_DIR = "data/vector_db/"

# ✅ Optionally rebuild from scratch (recommended after merging datasets)
if os.path.exists(DB_DIR):
    print(f"🧹 Removing old vector DB at {DB_DIR} ...")
    shutil.rmtree(DB_DIR)

os.makedirs(DB_DIR, exist_ok=True)

# --------------------------------------------------------------
# Step 2. Load and Clean Data
# --------------------------------------------------------------
print(f"📂 Loading data from {DATA_PATH} ...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} entries from JSON file.")

# 🧼 Clean + deduplicate questions
seen = set()
cleaned_data = []

for item in tqdm(data, desc="🧽 Cleaning data"):
    q = item.get("question", "").strip()
    a = item.get("answer", "").strip()

    if not q or not a:
        continue

    # Normalize whitespace and punctuation
    q = " ".join(q.split())
    a = " ".join(a.split())

    # Remove duplicates
    if q.lower() in seen:
        continue
    seen.add(q.lower())

    cleaned_data.append({"question": q, "answer": a})

print(f"✅ {len(cleaned_data)} unique Q&A pairs after cleaning.")

# --------------------------------------------------------------
# Step 3. Combine Questions + Answers
# --------------------------------------------------------------
documents = []
for item in cleaned_data:
    text = f"Question: {item['question']}\nAnswer: {item['answer']}"
    documents.append(text)

print(f"📄 Prepared {len(documents)} text blocks for embedding.")

# --------------------------------------------------------------
# Step 4. Split Text into Smaller Chunks (for better retrieval)
# --------------------------------------------------------------
'''splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # slightly larger for legal sections
    chunk_overlap=50,   # maintain context continuity
    length_function=len
)
chunks = splitter.create_documents(documents)
print(f"🧩 Split into {len(chunks)} smaller chunks for embedding.")''' '''//Issue : Adjust Chunking Logic'''

chunks = [Document(page_content=text) for text in documents]
print(f"🧩 Keeping {len(chunks)} Q&A pairs as single chunks.")


# --------------------------------------------------------------
# Step 5. Generate Embeddings
# --------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("⚙️ Generating embeddings and building vector store...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=DB_DIR
)

# --------------------------------------------------------------
# Step 6. Persist Vector Database
# --------------------------------------------------------------
vectorstore.persist()
print(f"💾 Vector database successfully saved to: {DB_DIR}")

# --------------------------------------------------------------
# Step 7. Test Retrieval
# --------------------------------------------------------------
query = "What is IPC Section 420?"
docs = vectorstore.similarity_search(query, k=2)

print("\n🔍 Sample Retrieval Test:")
for i, doc in enumerate(docs, 1):
    print(f"\nResult {i}:")
    print(doc.page_content[:300], "...")
