from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None  # lazy load — don't load on import!


def get_model():
    global _model
    if _model is None:
        print("Loading SentenceTransformer model...", flush=True)
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded!", flush=True)
    return _model


def get_embeddings(chunks: list[dict]) -> tuple[list[dict], np.ndarray]:
    texts = [f"File: {c['path']}\n\n{c['content']}" for c in chunks]
    embeddings = get_model().encode(texts, batch_size=32, show_progress_bar=True)
    return chunks, np.array(embeddings)