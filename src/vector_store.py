import numpy as np
import faiss


def create_vector_store(embeddings):
    """Build a FAISS index from the given embeddings.

    We use IndexFlatIP (inner product) because our embeddings are L2-normalized,
    which means inner product == cosine similarity. Simple and exact — no
    approximation needed at this dataset size.
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def search(query_embedding, index, top_k=5):
    """Search the index and return (scores, indices).

    Scores are cosine similarities in [0, 1] since vectors are normalized.
    """
    q = np.array(query_embedding, dtype=np.float32)
    scores, indices = index.search(q, top_k)
    return scores[0], indices[0]
