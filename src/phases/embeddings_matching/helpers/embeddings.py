"""
SBERT embedder utilities.

Path: src/phases/embeddings_matching/helpers/embeddings.py

Model: all-MiniLM-L6-v2 (default, overridable via SBERT_MODEL env var)
Provides:
 - load_sbert()
 - embed_texts()
 - cosine_sim_matrix()
 - text_hash()
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import hashlib
import os
from typing import Iterable, List

_MODEL_NAME_DEFAULT = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")

# Module level cached model instance
_model: SentenceTransformer = None


def load_sbert(model_name: str = _MODEL_NAME_DEFAULT) -> SentenceTransformer:
    """
    Lazily load and return a SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def _normalize_embeddings(embs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize embeddings row-wise.
    """
    if embs.size == 0:
        return embs
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return embs / norms


def embed_texts(
    texts: Iterable[str], model: SentenceTransformer = None, batch_size: int = 64
) -> np.ndarray:
    """
    Encode a list (or iterable) of strings and return L2-normalized numpy embeddings.
    Returns shape (len(texts), dim). If texts is empty, returns an array of shape (0, dim).
    """
    if model is None:
        model = load_sbert()
    texts = list(texts)
    dim = model.get_sentence_embedding_dimension()
    if len(texts) == 0:
        return np.zeros((0, dim), dtype=np.float32)
    embs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )
    embs = _normalize_embeddings(embs)
    return embs.astype(np.float32)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between two arrays of normalized embeddings.
    a: (n, d), b: (m, d) => returns (n, m)
    If either array is empty, returns an appropriately shaped zero array.
    """
    if a is None or b is None:
        return np.zeros((0, 0), dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.matmul(a, b.T).astype(np.float32)


def text_hash(s: str) -> str:
    """
    Deterministic sha256 hash of a string (useful for caching).
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
