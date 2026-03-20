from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_chunks(question, index, texts, top_k=4):

    q_embedding = model.encode([question])

    distances, indices = index.search(np.array(q_embedding), top_k)

    results = [texts[i] for i in indices[0]]

    return results


def generate_answer(question, context_chunks):

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a crystallography assistant.

Answer ONLY using the information provided in the context below.

If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Do NOT invent information.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="phi",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]