"""
--------------------------------------------------------------
  Streamlit Legal RAG Engine (Working Version = CLI Compatible)
--------------------------------------------------------------

- Same retrieval behavior as CLI version
- k = 8 (critical)
- Same embeddings & DB loading
- Returns ONE clean answer
--------------------------------------------------------------
"""

import os
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------
# Global Initialization
# --------------------------------------------------------------

DB_DIR = "data/vector_db/"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model
)

# IMPORTANT — use SAME k as CLI
retriever = db.as_retriever(search_kwargs={"k": 8})

# --------------------------------------------------------------
# Clean + Strict Prompt
# --------------------------------------------------------------

prompt_template = ChatPromptTemplate.from_template("""
You are a helpful Indian legal assistant.

Answer using ONLY the provided context.
If the context does not contain the answer, say:
"The provided context does not have enough information to answer precisely."

Rules:
- ONE clean answer only.
- No multiple questions.
- No bullet lists unless necessary.
- No repeating context.

Context:
{context}

Question:
{question}

Final Answer:
""")

# --------------------------------------------------------------
# Streamlit RAG Call
# --------------------------------------------------------------

def ask_streamlit(query: str) -> str:

    docs = retriever.invoke(query)
    if not docs:
        return "The provided context does not have enough information to answer precisely."

    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = prompt_template.format(
        context=context_text,
        question=query
    )

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful Indian legal assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=350,
        temperature=0.25
    )

    answer = response.choices[0].message["content"]

    clean = (
        answer.replace("Final Answer:", "")
              .replace("Answer:", "")
              .replace("[/INST]", "")
              .strip()
    )

    return clean
