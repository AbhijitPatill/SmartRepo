from sentence_transformers import SentenceTransformer
import numpy as np

# Switched from all-mpnet-base-v2 (420MB, slow) to all-MiniLM-L6-v2 (80MB, 6x faster)
# Quality difference is minimal for code Q&A
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

def get_embeddings(chunks: list[dict]) -> tuple[list[dict], np.ndarray]:
    texts = [f"File: {c['path']}\n\n{c['content']}" for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return chunks, np.array(embeddings)