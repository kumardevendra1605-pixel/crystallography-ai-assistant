from src.docx_loader import load_documents
from src.text_chunker import chunk_documents
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.rag_pipeline import retrieve_chunks, generate_answer

# Load documents
docs = load_documents("data")

# Chunk text
chunks = chunk_documents(docs)

# Create embeddings
texts, embeddings = create_embeddings(chunks)

# Create FAISS vector database
index = create_vector_store(embeddings)

print("documents:", len(docs))
print("chunks:", len(chunks))
print("embeddings:", len(embeddings))

# Ask question
question = "What causes weak diffraction in crystallography?"

retrieved_chunks = retrieve_chunks(question, index, texts)

answer = generate_answer(question, retrieved_chunks)

print("\nAI Answer:\n")
print(answer)