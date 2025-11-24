import streamlit as st
from src.rag.final_rag import ask_streamlit
import re

# --------------------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------------------
st.set_page_config(
    page_title="NyAI Legal Assistant",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ NyAI Legal Assistant")
st.write("Ask any legal question.")

# --------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------------------
# Chat Display
# --------------------------------------------------------------
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])


# --------------------------------------------------------------
# User Input
# --------------------------------------------------------------
user_query = st.chat_input("Ask your legal question...")

if user_query:

    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # ----------------------------------------------------------
    # Query RAG Model
    # ----------------------------------------------------------
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documents..."):
            answer = ask_streamlit(user_query)

        # ------------------------------------------------------
        # CLEAN UP MODEL OUTPUT
        # ------------------------------------------------------
        answer = re.sub(r"Question:\s*.*", "", answer, flags=re.IGNORECASE).strip()
        answer = re.sub(r"^Q[:\-].*", "", answer).strip()
        answer = "\n".join([line for line in answer.split("\n") if line.strip()])

        st.write(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
