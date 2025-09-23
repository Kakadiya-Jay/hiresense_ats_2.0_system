# scoring.py
"""
Combine BM25 and semantic matching into a simple normalized scoring mechanism.

Principles:
 - BM25 gives relevance by token overlap (keyword match).
 - Semantic similarity gives meaning-level match.
 - We combine them in weighted fashion: score = w_bm25 * norm_bm25 + w_sem * norm_sem
 - Also include simple heuristics: presence of required keywords or min experience years (if provided).
"""
from typing import List, Dict, Tuple
import numpy as np


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    minv, maxv = arr.min(), arr.max()
    if maxv - minv == 0:
        return [1.0 for _ in arr]
    norm = ((arr - minv) / (maxv - minv)).tolist()
    return norm


def combine_scores(
    bm25_pairs: List[Tuple[int, float]],
    sem_pairs: List[Tuple[int, float]],
    w_bm25: float = 0.5,
    w_sem: float = 0.5,
    top_k: int = 5,
) -> List[Dict]:
    """
    bm25_pairs: list of (idx, score) from BM25 (top_k, sorted)
    sem_pairs: list of (idx, score) from semantic matcher (top_k, sorted)
    Returns list of dicts: {idx, bm25_score, sem_score, combined_score}
    """
    # Build maps
    bm_map = {idx: sc for idx, sc in bm25_pairs}
    sem_map = {idx: sc for idx, sc in sem_pairs}
    all_idxs = list(set(list(bm_map.keys()) + list(sem_map.keys())))
    bm_list = [bm_map.get(i, 0.0) for i in all_idxs]
    sem_list = [sem_map.get(i, 0.0) for i in all_idxs]

    nbm = normalize_scores(bm_list)
    nsem = normalize_scores(sem_list)

    results = []
    for i, idx in enumerate(all_idxs):
        combined = w_bm25 * nbm[i] + w_sem * nsem[i]
        results.append(
            {
                "idx": idx,
                "bm25_raw": bm_map.get(idx, 0.0),
                "sem_raw": sem_map.get(idx, 0.0),
                "bm25_norm": nbm[i],
                "sem_norm": nsem[i],
                "score": float(combined),
            }
        )
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    return results_sorted[:top_k]


# quick demo
if __name__ == "__main__":
    bm = [(0, 10), (1, 5)]
    sem = [(1, 0.8), (2, 0.4)]
    print(combine_scores(bm, sem))
