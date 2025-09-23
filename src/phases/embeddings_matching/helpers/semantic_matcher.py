# semantic_matcher.py
"""
Semantic matching using sentence-transformers (SBERT).
If model is unavailable, you can fallback to TF-IDF similarity using scikit-learn.
"""
from typing import List, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util

    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMatcher:
    def __init__(self, texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
        """
        texts: list of strings (documents or sentences) that form the corpus
        """
        self.texts = texts
        self.model_name = model_name
        if SBERT_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            self.embeddings = self.model.encode(
                self.texts, convert_to_tensor=True, show_progress_bar=False
            )
            self.use = "sbert"
        else:
            self.vectorizer = TfidfVectorizer().fit(self.texts)
            self.tfidf = self.vectorizer.transform(self.texts)
            self.use = "tfidf"

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.use == "sbert":
            q_emb = self.model.encode(query_text, convert_to_tensor=True)
            sims = util.cos_sim(q_emb, self.embeddings).cpu().numpy().flatten()
        else:
            q_vec = self.vectorizer.transform([query_text])
            sims = cosine_similarity(q_vec, self.tfidf).flatten()
        ranked_idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in ranked_idx]


# demo
if __name__ == "__main__":
    txts = [
        "I love python programming",
        "Experience in data science",
        "Worked on NLP projects",
    ]
    sem = SemanticMatcher(txts)
    print(sem.query("NLP and python", top_k=3))
