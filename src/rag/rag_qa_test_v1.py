"""
Offline RAG Legal Q&A Test - No Internet, No HF Token Required
"""

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline

DB_DIR = "vector_db/"

# Load embeddings and vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load local model for text generation
device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text2text-generation", model="google/flan-t5-base", device=device)

prompt = ChatPromptTemplate.from_template("""
You are a helpful legal assistant. Use the provided context to answer the user's question clearly and accurately.
If you don't know, say "I am not certain based on the available context."

Context:
{context}

Question:
{question}

Answer:
""")

def ask_legal_question(query: str):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted = prompt.format(context=context, question=query)
    output = generator(formatted, max_length=256, temperature=0.3)[0]["generated_text"]

    print(f"\n🧑‍⚖️ Question: {query}\n💬 Answer: {output}\n")
    print("📚 Context used:\n")
    for i, doc in enumerate(docs, 1):
        print(f"Source {i}:\n{doc.page_content[:350]}...\n")

if __name__ == "__main__":
    print("\n⚖️ Testing Offline RAG Q&A...\n")
    ask_legal_question("What does Article 21 of the Constitution of India guarantee?")
    ask_legal_question("What is Section 420 of the IPC?")
