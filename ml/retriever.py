import chromadb
import numpy as np
import os
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "..", "data", "chromadb")
os.makedirs(CHROMA_DIR, exist_ok=True)

COLLECTION_NAME = "smartrepo_active"

# Lazy init — don't create on import
_client = None
_bm25_index = None
_bm25_chunks = []


def get_client():
    global _client
    if _client is None:
        print("Initializing ChromaDB client...", flush=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        print("ChromaDB ready!", flush=True)
    return _client


def get_collection():
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def build_index(chunks: list[dict], embeddings: np.ndarray):
    global _bm25_index, _bm25_chunks

    try:
        get_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection()

    ids       = [str(i) for i in range(len(chunks))]
    documents = [c["content"] for c in chunks]
    metadatas = [{"path": c["path"], "extension": c.get("extension", "")} for c in chunks]
    embeds    = embeddings.tolist()

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids        = ids[i:i+batch_size],
            embeddings = embeds[i:i+batch_size],
            documents  = documents[i:i+batch_size],
            metadatas  = metadatas[i:i+batch_size]
        )

    _bm25_chunks = chunks
    tokenized = [_tokenize(c["path"] + " " + c["content"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized)

    print(f"[Hybrid] Indexed {len(chunks)} chunks — ChromaDB + BM25 ready", flush=True)


def _tokenize(text: str) -> list[str]:
    import re
    text = text.lower()
    tokens = re.split(r'[\s\.\(\)\{\}\[\]:,;=<>/"\'\\]+', text)
    return [t for t in tokens if len(t) > 1]


def _load_bm25_from_chroma():
    global _bm25_index, _bm25_chunks
    try:
        collection = get_client().get_collection(COLLECTION_NAME)
        results = collection.get(include=["documents", "metadatas"])
        chunks = [
            {"path": m["path"], "content": d, "extension": m.get("extension", "")}
            for d, m in zip(results["documents"], results["metadatas"])
        ]
        _bm25_chunks = chunks
        tokenized = [_tokenize(c["path"] + " " + c["content"]) for c in chunks]
        _bm25_index = BM25Okapi(tokenized)
        print(f"[Hybrid] BM25 rebuilt from ChromaDB — {len(chunks)} chunks", flush=True)
    except Exception as e:
        print(f"[Hybrid] BM25 rebuild failed: {e}", flush=True)


def search(query_embedding: np.ndarray, query_text: str = "", k: int = 7) -> list[dict]:
    global _bm25_index

    if _bm25_index is None:
        _load_bm25_from_chroma()

    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    fetch_k = min(k * 2, total)

    semantic_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

    semantic_scores = {}
    for idx, (doc, meta, dist) in enumerate(zip(
        semantic_results["documents"][0],
        semantic_results["metadatas"][0],
        semantic_results["distances"][0]
    )):
        chunk_id = semantic_results["ids"][0][idx]
        similarity = 1 - dist
        semantic_scores[chunk_id] = {
            "score": similarity,
            "chunk": {"path": meta["path"], "content": doc, "extension": meta.get("extension", "")}
        }

    bm25_scores = {}
    if query_text and _bm25_index is not None:
        tokens = _tokenize(query_text)
        scores = _bm25_index.get_scores(tokens)
        max_score = max(scores) if max(scores) > 0 else 1
        top_indices = np.argsort(scores)[::-1][:fetch_k]

        for idx in top_indices:
            if idx < len(_bm25_chunks):
                chunk_id = str(idx)
                bm25_scores[chunk_id] = {
                    "score": float(scores[idx]) / max_score,
                    "chunk": _bm25_chunks[idx]
                }

    all_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())
    merged = {}
    for cid in all_ids:
        sem   = semantic_scores.get(cid, {}).get("score", 0)
        bm25  = bm25_scores.get(cid, {}).get("score", 0)
        chunk = (semantic_scores.get(cid) or bm25_scores.get(cid))["chunk"]
        merged[cid] = {
            "score": 0.6 * sem + 0.4 * bm25,
            "chunk": chunk
        }

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return [r["chunk"] for r in ranked[:k]]


def index_exists() -> bool:
    try:
        collection = get_client().get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


def get_indexed_files() -> list[str]:
    try:
        collection = get_client().get_collection(COLLECTION_NAME)
        results = collection.get(include=["metadatas"])
        seen = []
        for meta in results["metadatas"]:
            path = meta.get("path", "")
            if path and path not in seen:
                seen.append(path)
        return seen
    except Exception:
        return []