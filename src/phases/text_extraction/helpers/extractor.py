# src/phases/text_extraction/helpers/extractor.py
"""
Extract text and basic metadata from PDF using PyMuPDF + pytesseract fallback.
Public functions:
- extract_text_and_meta(pdf_path) -> dict
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
from typing import Dict, List

URL_REGEX = re.compile(
    r"(https?://[^\s\)\]\>]+|www\.[^\s\)\]\>]+)", flags=re.IGNORECASE
)
EMAIL_REGEX = re.compile(
    r"([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)", flags=re.IGNORECASE
)


def ocr_page_to_text(page, dpi: int = 200) -> str:
    """OCR a page with pytesseract (fallback)."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def fix_hyphenation_and_linebreaks(text: str) -> str:
    """
    Fix typical PDF hyphenation and normalize linebreaks.
    Examples:
      "optimi-\n zation" -> "optimization"
      "en\n tire" -> "en tire"
    """
    text = re.sub(r"-\s*\n\s*", "", text)  # join hyphenated EOL words
    text = re.sub(
        r"(\w)\s*\n\s*(\w)", r"\1 \2", text
    )  # join broken words across lines (keep space)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def extract_text_and_meta(pdf_path: str) -> Dict:
    """
    Extract text + simple metadata:
      - full_text (joined pages)
      - pages (list)
      - links (raw list from annotations + inline urls: [{'url','anchor_text', 'page'}])
      - emails (list)
      - phones (simple heuristic list)
      - cleaned_text (initial; hyphenation fixed)
    """
    doc = fitz.open(pdf_path)
    pages_text: List[str] = []
    raw_links: List[Dict] = []

    for p in doc:
        text = p.get_text("text")
        if not text.strip():
            text = ocr_page_to_text(p)
        text = fix_hyphenation_and_linebreaks(text)
        pages_text.append(text)

        # link annotations
        try:
            for l in p.get_links():
                if l.get("uri"):
                    raw_links.append(
                        {"url": l.get("uri"), "anchor_text": None, "page": p.number}
                    )
        except Exception:
            # non-fatal: skip annotation extraction errors
            pass

    full_text = "\n".join(pages_text)

    # inline URLs discovered via regex
    inline_urls = list(set([u.rstrip(".,;:") for u in URL_REGEX.findall(full_text)]))
    for u in inline_urls:
        raw_links.append({"url": u, "anchor_text": None, "page": None})

    # emails inline
    emails = list({m.group(1).lower() for m in EMAIL_REGEX.finditer(full_text)})

    # simple phone heuristic
    raw_phones = re.findall(
        r"(?:\+\d{1,3}[-\s]?)?(?:\(\d{2,4}\)[-\s]?)?\d{5,12}", full_text
    )
    phones = list({p for p in raw_phones if len(re.sub(r"\D", "", p)) >= 7})

    cleaned_text = full_text  # injection and label-cleaning will be applied in link classifier phase

    return {
        "full_text": full_text,
        "pages": pages_text,
        "links": raw_links,
        "emails": emails,
        "phones": phones,
        "cleaned_text": cleaned_text,
    }
