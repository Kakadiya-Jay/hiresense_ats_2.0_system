# preprocess.py
"""
spaCy-based preprocessing:
 - sentence segmentation
 - tokenization (lemmas, remove stopwords/punct)
 - light section heuristics (detect 'Education', 'Experience', 'Projects' headers)
"""
from typing import List, Dict, Tuple
import spacy
import re

nlp = spacy.load("en_core_web_sm", disable=["ner"])  # keep lightweight

SECTION_HEADERS = [
    "education",
    "experience",
    "work experience",
    "projects",
    "skills",
    "certifications",
    "achievements",
    "summary",
    "contact",
    "objective",
]


def simple_section_split(text: str) -> Dict[str, str]:
    """
    Very simple: look for SECTION HEADERS (case-insensitive) and split into sections.
    Returns a dict: {section_name: text}
    """
    # Normalize and find header indices
    lower = text.lower()
    # Build a regex to find headers at line starts
    pattern = (
        r"(?m)^(?P<header>"
        + "|".join(re.escape(h) for h in SECTION_HEADERS)
        + r")\b.*$"
    )
    headers = list(re.finditer(pattern, lower, flags=re.IGNORECASE))
    if not headers:
        return {"full": text.strip()}

    sections = {}
    # Add start and end
    indices = [(m.start(), m.group("header")) for m in headers]
    indices_sorted = sorted(indices, key=lambda x: x[0])
    for idx, (pos, header) in enumerate(indices_sorted):
        start = pos
        current_header = header.strip()
        next_start = (
            indices_sorted[idx + 1][0] if idx + 1 < len(indices_sorted) else len(text)
        )
        sec_text = text[start:next_start].strip()
        sections[current_header] = sec_text
    # Add fallback: full
    sections["full"] = text.strip()
    return sections


def clean_sentence(sent):
    # Remove multiple spaces/newlines, stray bullets
    s = re.sub(r"[\u2022•\t]+", " ", sent)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sentences_from_text(text: str) -> List[str]:
    doc = nlp(text)
    sents = [clean_sentence(sent.text) for sent in doc.sents if sent.text.strip()]
    return sents


def tokens_for_sentence(sentence: str, lemmatize=True, min_len=2) -> List[str]:
    doc = nlp(sentence)
    tokens = []
    for t in doc:
        if t.is_stop or t.is_punct or t.is_space:
            continue
        lemma = t.lemma_.lower() if lemmatize else t.text.lower()
        if len(lemma) >= min_len:
            tokens.append(lemma)
    return tokens


def preprocess_for_bm25(text: str) -> List[List[str]]:
    """
    Returns a list of token lists (one per sentence) for BM25 indexing.
    """
    sents = sentences_from_text(text)
    return [tokens_for_sentence(s) for s in sents if tokens_for_sentence(s)]


# quick demo
if __name__ == "__main__":
    sample = "Education\nB.Tech in CS 2019-2023\nExperience\nSoftware Intern at X."
    print(simple_section_split(sample))
    print(sentences_from_text(sample))
    print(preprocess_for_bm25(sample))
