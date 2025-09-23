# bm25_matcher.py
"""
BM25 matcher using rank_bm25.
Index a corpus (list of documents or list of sentences) and query with tokens.
"""
from typing import List, Tuple
from rank_bm25 import BM25Okapi


class BM25Matcher:
    def __init__(self, tokenized_corpus: List[List[str]]):
        """
        tokenized_corpus: list of token lists (documents or sentences)
        """
        self.corpus = tokenized_corpus
        self.bm25 = BM25Okapi(self.corpus)

    def query(
        self, tokenized_query: List[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Returns list of (index_in_corpus, score) sorted by score desc.
        """
        scores = self.bm25.get_scores(tokenized_query)
        scored = [(i, float(score)) for i, score in enumerate(scores)]
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        return scored_sorted[:top_k]


# demo
if __name__ == "__main__":
    corpus = [["machine", "learning"], ["data", "science"], ["python", "developer"]]
    bm = BM25Matcher(corpus)
    print(bm.query(["python"]))
