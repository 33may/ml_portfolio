from typing import List, Dict
import numpy as np
from loguru import logger
from openai import OpenAI


CHAT_MODEL = "gpt-4o-2024-08-06"
EMB_MODEL = "text-embedding-3-small"


client = OpenAI(
    api_key=openai_key
)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 200,
) -> List[str]:
    """
    Slide a fixed-size window across text to create overlapping chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    logger.debug(f"Created {len(chunks)} text chunks")
    return chunks


def build_embeddings(
    chunks: List[str], embed_model: str = EMB_MODEL
) -> List[Dict]:
    """Compute embeddings for each chunk and return [{text, vector}, …]."""
    logger.info("Computing embeddings for chunks")
    resp = client.embeddings.create(model=embed_model, input=chunks)
    vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return [{"text": chunks[i], "vector": vectors[i]} for i in range(len(chunks))]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between row vectors in `a` and `b`.
    a: (1, d)
    b: (N, d)
    returns: (1, N)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)

def similarity_search(
    query: str,
    kb: List[Dict],
    top_k: int = 5,
) -> List[str]:
    """
    Return `top_k` chunk texts from `kb` most similar to `query`.
    """
    q_vec = build_embeddings([query])[0]
    mats = np.vstack([item["vector"] for item in kb])
    sims = cosine_similarity(np.expand_dims(q_vec["vector"], axis=0), mats)
    idx = sims.argsort()[-top_k:][::-1]
    return [kb[i]["text"] for i in idx[0][:top_k]]