# src/phases/feature_extraction/helpers/phrase_extractor.py
from typing import List, Tuple, Union
import spacy

# load small model, disable unneeded components for speed
nlp = spacy.load("en_core_web_sm", disable=["textcat"])


def _coerce_to_text(value: Union[str, list, dict, None]) -> str:
    """
    Convert different section representations into a single string.
    - str -> returned as-is
    - list -> join elements with double newline (preserves block boundaries)
    - dict -> join values (order not guaranteed) with double newline
    - None -> empty string
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Join nested lists/dicts recursively
        parts = []
        for item in value:
            if isinstance(item, (list, dict)):
                parts.append(_coerce_to_text(item))
            else:
                parts.append(str(item))
        return "\n\n".join([p for p in parts if p])
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(str(k) + ": " + _coerce_to_text(v))
        return "\n\n".join([p for p in parts if p])
    # fallback
    return str(value)


def sentence_and_ngram_candidates(
    text: Union[str, list, dict, None], max_n=3
) -> Tuple[List[str], List[str]]:
    """
    Return sentence-level texts and candidate n-grams (1..max_n) filtered by POS patterns.
    Accepts text as str, list, dict, or None. Internally coerces to a single string before spaCy.
    """
    coerced_text = _coerce_to_text(text)
    if not coerced_text.strip():
        return [], []

    doc = nlp(coerced_text)
    sentences = [sent.text.strip() for sent in doc.sents]

    tokens = [t for t in doc if not t.is_space]
    candidates = set()
    for i in range(len(tokens)):
        for n in range(1, max_n + 1):
            if i + n <= len(tokens):
                span = tokens[i : i + n]
                pos_tags = [t.pos_ for t in span]
                # build candidate text
                text_candidate = span[0].text
                for t in span[1:]:
                    text_candidate += " " + t.text
                # basic filter: at least one NOUN/PROPN and not all stopwords/punct
                if any(p in ("NOUN", "PROPN") for p in pos_tags):
                    if len(text_candidate) > 1 and not all(
                        t.is_stop or t.is_punct for t in span
                    ):
                        candidates.add(text_candidate.strip())
    return sentences, list(candidates)
