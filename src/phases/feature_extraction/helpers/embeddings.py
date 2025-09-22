# src/phases/feature_extraction/helpers/embeddings.py
"""
Embeddings helper using SentenceTransformers with defensive coercion.

Functions:
 - get_model(name)
 - embed_texts(texts) -> numpy array of normalized embeddings
 - cosine_similarity(a, b) -> dot-product similarity (assumes normalized embeddings)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import Iterable, List

logger = logging.getLogger(__name__)

_model = None


def get_model(name: str = "all-mpnet-base-v2"):
    """
    Lazy-load the SBERT model. Choose a lighter model like 'all-MiniLM-L6-v2'
    for low latency if needed.
    """
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(name)
        except Exception as e:
            logger.exception("Failed to load SentenceTransformer model %s: %s", name, e)
            raise
    return _model


def _coerce_texts(texts: Iterable) -> List[str]:
    """
    Ensure texts is a list of non-empty strings. Coerce non-strings to str.
    """
    if texts is None:
        return []
    out = []
    for t in texts:
        if t is None:
            continue
        if isinstance(t, str):
            s = t.strip()
        else:
            try:
                s = str(t).strip()
            except Exception:
                s = ""
        if s:
            out.append(s)
    return out


def embed_texts(texts: Iterable, model_name: str = "all-mpnet-base-v2"):
    """
    Embed a list of texts. Returns a numpy array of shape (n, d).
    If texts is empty, returns an empty numpy array with shape (0,).
    """
    safe_texts = _coerce_texts(texts)
    if not safe_texts:
        return np.array([])
    model = get_model(model_name)
    try:
        emb = model.encode(safe_texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb
    except Exception as e:
        logger.exception("Embedding failed: %s", e)
        raise


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    Compute cosine similarity between a (m,d) and b (n,d) and return (m,n) matrix.
    Accepts 1-D arrays for convenience.
    """
    if a is None or b is None:
        raise ValueError("Inputs to cosine_similarity must not be None")
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        # return empty similarity arrays with appropriate shape
        if a.ndim == 1 and b.ndim == 1:
            return np.array([[]])
        if a.ndim == 1:
            return np.zeros((1, b.shape[0]))
        return np.zeros((a.shape[0], b.shape[0]))
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    try:
        sim = np.dot(a, b.T)
        return sim
    except Exception as e:
        logger.exception("cosine_similarity failed: %s", e)
        raise
