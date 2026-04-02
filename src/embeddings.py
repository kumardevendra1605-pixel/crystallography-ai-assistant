import os
import hashlib
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

# Store the cache next to the database folder, relative to this file's location
_project_root = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(_project_root, "database")
CACHE_FILE = os.path.join(CACHE_DIR, "qa_embeddings.pkl")

# Lazy-loaded model — only loads when first needed
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    """Encode a list of strings into normalized embeddings."""
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query):
    """Encode a single query. Returns shape (1, 384)."""
    return get_model().encode([query], normalize_embeddings=True)


def _hash_content(qa_pairs):
    """SHA256 over all question+answer text so we know when the data changed."""
    h = hashlib.sha256()
    for p in qa_pairs:
        h.update((p["question"] + p["answer"]).encode("utf-8"))
    return h.hexdigest()


def build_and_cache_embeddings(qa_pairs):
    """Build embeddings for the whole dataset and cache them to disk.

    We embed question + the first 300 chars of the answer together — this
    gives much better retrieval than question-only because the answer text
    often contains the key terms people search for.

    The cache is invalidated by a content hash, so adding or editing any
    Q&A pair automatically triggers a rebuild on the next startup.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    current_hash = _hash_content(qa_pairs)

    # Check if we have a valid cache already
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("content_hash") == current_hash:
                logger.info("Loaded embeddings from cache.")
                return cached["embeddings"]
            logger.info("Data changed — rebuilding embeddings.")
        except Exception as e:
            logger.warning(f"Cache was corrupt ({e}), rebuilding from scratch.")
            os.remove(CACHE_FILE)

    # Build fresh embeddings
    texts = [p["question"] + " " + p["answer"][:300] for p in qa_pairs]
    embeddings = embed_texts(texts)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"content_hash": current_hash, "embeddings": embeddings}, f)

    logger.info(f"Cached {len(embeddings)} embeddings to disk.")
    return embeddings
