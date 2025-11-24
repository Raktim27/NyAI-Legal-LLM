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

Issue : model tends to “hallucinate” or fall back on its internal knowledge base when asked about any ireevant information not present in the retrieved documents.
--------------------------------------------------------------
"""

import os
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------
# Step 1. Setup and Token
# --------------------------------------------------------------
DB_DIR = "data/vector_db/"  # Your existing Chroma persistence directory

# ✅ Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your actual token

# ✅ Initialize HF client for Zephyr chat model
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# --------------------------------------------------------------
# Step 2. Load Vector DB and Embeddings
# --------------------------------------------------------------
print("🚀 Loading Chroma DB and embedding model...")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 8})

print("✅ Vector Database loaded successfully!")

# --------------------------------------------------------------
# Step 3. Define Prompt Template
# --------------------------------------------------------------
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
# Step 4. Define Function to Query Model
# --------------------------------------------------------------
def ask_legal_question(query: str):
    print(f"\n🧑‍⚖️ User Query: {query}\n")

    # Retrieve top relevant documents
    docs = retriever.invoke(query)
    if not docs:
        print("⚠️ No relevant documents found in the database.")
        return

    # Combine context
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Create full prompt
    prompt = prompt_template.format(context=context_text, question=query)

    print("💬 Querying Zephyr model...")

    # Generate answer (using modern chat_completion API)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful Indian legal assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.3
    )

    # Extract and display model output
    answer = response.choices[0].message["content"]

    print("\n📜 Model Answer:\n", answer)
    print("\n📚 Context Sources Used:")
    for i, doc in enumerate(docs, 1):
        print(f"\nSource {i}:\n{doc.page_content[:300]}...")


# --------------------------------------------------------------
# Step 5. Interactive Loop
# --------------------------------------------------------------
if __name__ == "__main__":
    print("\n⚖️ Welcome to NyAI Legal RAG Assistant (Zephyr-powered)")
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


# --------------------------------------------------------------
# EXPORT FUNCTION FOR STREAMLIT (wrapper)
# --------------------------------------------------------------
def ask(query: str):
    # Retrieve top relevant documents
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found in the database."

    # Combine context
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Create full prompt
    prompt = prompt_template.format(context=context_text, question=query)

    # Query model
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful Indian legal assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.3
    )

    answer = response.choices[0].message["content"].strip()
    return answer

