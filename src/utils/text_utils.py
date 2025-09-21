# src/utils/text_utils.py

import re
import string


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces/newlines into single space."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower() if text else ""


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from the given text.
    Keeps alphanumeric characters, whitespace, and some structural symbols like '+' and '|'
    that are often used in resumes for skills/tech stacks.
    """
    if not text:
        return ""

    # Define allowed symbols
    allowed = set("+|@#")
    cleaned_chars = []
    for char in text:
        if char.isalnum() or char.isspace() or char in allowed:
            cleaned_chars.append(char)
        # else drop the punctuation
    return "".join(cleaned_chars)


def clean_text_pipeline(text: str) -> str:
    """
    Apply a standard cleaning pipeline:
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text
