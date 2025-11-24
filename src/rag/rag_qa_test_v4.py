"""
--------------------------------------------------------------
 Interactive Legal RAG Assistant (LangChain + Zephyr)
--------------------------------------------------------------
Author: Raktim
Purpose:
    - Load existing Chroma vector DB
    - Retrieve top legal documents
    - Use Hugging Face Zephyr-7B (chat model) for answer generation
    - Interactive chat mode for real-time Q&A

Issue with v3: model tends to “hallucinate” or fall back on its internal knowledge base when asked about any ireevant information not present in the retrieved documents.
Solution in v4: Updated prompt to explicitly instruct the model to only use the provided context for answering questions , adding Guardrails to avoid hallucination.

Key Fixes:
    ✅ Softer prompt (Zephyr now answers when context is relevant)
    ✅ Shorter context preview
    ✅ Reduced hallucinations on non-legal queries
--------------------------------------------------------------
"""
import os
import numpy as np
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------
# Setup
# --------------------------------------------------------------
DB_DIR = "data/vector_db/"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

print("✅ Vector Database loaded successfully!")

# --------------------------------------------------------------
# Prompt Template (Smarter Logic)
# --------------------------------------------------------------
prompt_template = ChatPromptTemplate.from_template("""
You are an expert Indian legal assistant specializing in Indian Penal Code (IPC).

You will be given:
- Legal context retrieved from a verified database
- A user’s question

### Guidelines:
1. Use the **provided context only** to answer.
2. If the context is not exactly the same but clearly relevant, answer based on it.
3. If the context is completely unrelated, reply:
   "The provided context does not have enough information to answer precisely."
4. Be formal, factual, and concise.
5. Do not mention that you are an AI.

Examples:
Q: What happens if I commit theft in a house?
Context mentions: “theft in a building, tent or vessel.”
✅ A: Such an offence is punishable under IPC Section 380 with imprisonment up to seven years and a fine.

Q: What is Sales Cloud?
Context unrelated.
❌ A: The provided context does not have enough information to answer precisely.

Now answer the user’s question strictly based on the following context:

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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_answer(text):
    """Extract only the answer portion from stored Q&A chunks."""
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()

# --------------------------------------------------------------
# Main RAG Function
# --------------------------------------------------------------
def ask_legal_question(query: str):
    print(f"\n🧑‍⚖️ User Query: {query}\n")

    results = db.similarity_search_with_score(query, k=8)
    if not results:
        print("⚠️ No relevant documents found.")
        return

    q_emb = embedding_model.embed_query(query)
    doc_texts = [extract_answer(doc.page_content) for doc, _ in results]
    doc_embs = embedding_model.embed_documents(doc_texts)
    cosines = [cosine_sim(q_emb, emb) for emb in doc_embs]
    max_cos = max(cosines)

    print(f"🔍 Max semantic similarity = {max_cos:.3f}")

    THRESHOLD = 0.5
    if max_cos < THRESHOLD:
        print("⚠️ Context not relevant enough (semantic < threshold).")
        print("💬 The provided context does not have enough information to answer precisely.")
        return

    # Use top 2 relevant chunks (answers only)
    top_docs = [doc_texts[i] for i in np.argsort(cosines)[-2:][::-1]]
    context_text = "\n\n".join(top_docs)

    prompt = prompt_template.format(context=context_text, question=query)

    print("💬 Querying Zephyr model...")

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful Indian legal assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.3
    )

    answer = response.choices[0].message["content"]

    print("\n📜 Model Answer:\n", answer)
    print("\n📚 Context Used:")
    for i, doc in enumerate(top_docs, 1):
        print(f"  🔹 Source {i}: {doc[:180]}...\n")

# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
if __name__ == "__main__":
    print("\n⚖️ Welcome to NyAI Legal RAG Assistant (Zephyr-powered v9)")
    print("Type your question below. Type 'exit' to quit.\n")

    while True:
        user_query = input("❓ Enter your legal question: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("\n👋 Exiting Legal Assistant. Goodbye!\n")
            break
        if not user_query:
            continue

        ask_legal_question(user_query)
        print("\n" + "-" * 70 + "\n")