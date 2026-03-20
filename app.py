import streamlit as st

from src.docx_loader import load_documents
from src.text_chunker import chunk_documents
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.rag_pipeline import retrieve_chunks, generate_answer


st.title("🔬 Crystallography AI Assistant")

# Load pipeline only once
@st.cache_resource
def setup_pipeline():

    docs = load_documents("data")
    chunks = chunk_documents(docs)
    texts, embeddings = create_embeddings(chunks)
    index = create_vector_store(embeddings)

    return index, texts


index, texts = setup_pipeline()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# User input
prompt = st.chat_input("Ask something about crystallography")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    chunks = retrieve_chunks(prompt, index, texts)

    answer = generate_answer(prompt, chunks)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})