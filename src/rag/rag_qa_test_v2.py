"""
--------------------------------------------------------------
 Test RAG Pipeline (Modern LangChain + Zephyr Conversational)
--------------------------------------------------------------
Author: Raktim
Purpose:
    - Load existing Chroma vector DB
    - Retrieve best-matching context
    - Use Hugging Face Zephyr conversational model (chat-based)
    - Compatible with latest huggingface_hub InferenceClient API
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
DB_DIR = "vector_db/"

# ✅ Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your token

# ✅ Initialize direct client for Zephyr chat model
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# --------------------------------------------------------------
# Step 2. Load Embeddings + Vector Database
# --------------------------------------------------------------
print("🚀 Loading Chroma DB and embedding model...")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 4})

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

    # Retrieve relevant context
    docs = retriever.invoke(query)
    if not docs:
        print("⚠️ No relevant context found in vector DB.")
        return

    context_text = "\n\n".join([doc.page_content for doc in docs])
    prompt = prompt_template.format(context=context_text, question=query)

    print("💬 Querying Zephyr chat model...\n")

    # ✅ Modern HF API: chat-based completion
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful Indian legal assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.3
    )

    # Extract the generated answer text
    answer = response.choices[0].message["content"]

    print("📜 Model Answer:\n", answer)
    print("\n📚 Context Sources Used:")
    for i, doc in enumerate(docs, 1):
        print(f"\nSource {i}:\n{doc.page_content[:300]}...")


# --------------------------------------------------------------
# Step 5. Test Queries
# --------------------------------------------------------------
if __name__ == "__main__":
    ask_legal_question("What does Article 21 of the Constitution of India guarantee?")
    ask_legal_question("What is Section 420 of the IPC?")
    ask_legal_question("What are the rights of an arrested person under CrPC?")
