"""Lecture et nettoyage des PDF.

PyMuPDF suffit pour des textes de loi (PDF à texte natif). Si ton PDF est un scan, le texte sera vide :
c'est le signal pour tester un parseur avec OCR (voir docs/TECH_CHOICES.md, section Parsing).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path


def extract_pages(pdf_path: str | Path) -> list[str]:
    import fitz  # PyMuPDF, importé ici pour que le reste du code se teste sans lui

    with fitz.open(str(pdf_path)) as doc:
        return [page.get_text("text") for page in doc]


def strip_repeated_lines(
    pages: list[str], threshold: float = 0.5, max_len: int = 80, min_pages: int = 4
) -> list[str]:
    """Supprime les en-têtes/pieds de page : lignes courtes présentes sur >= threshold des pages.

    Sans ça, "Bulletin officiel n° ..." ou "Code du travail" se retrouvent dans des centaines de chunks
    et polluent BM25 comme les embeddings.
    """
    if len(pages) < min_pages:
        return pages
    counts: Counter[str] = Counter()
    for p in pages:
        for line in {ln.strip() for ln in p.splitlines() if ln.strip()}:
            if len(line) <= max_len:
                counts[line] += 1
    repeated = {ln for ln, c in counts.items() if c / len(pages) >= threshold}
    cleaned = []
    for p in pages:
        kept = [ln for ln in p.splitlines() if ln.strip() not in repeated]
        cleaned.append("\n".join(kept))
    return cleaned


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"-\n(?=[a-zàâçéèêëîïôûùüÿ])", "", text)  # mot coupé en fin de ligne
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pages_to_text(pages: list[str]) -> tuple[str, list[int]]:
    """Concatène les pages et renvoie aussi l'offset de début de chaque page (pour citer la page)."""
    offsets, parts, pos = [], [], 0
    for p in pages:
        offsets.append(pos)
        parts.append(p)
        pos += len(p) + 1
    return "\n".join(parts), offsets


def load_document(pdf_path: str | Path) -> tuple[str, list[int]]:
    pages = [clean_text(p) for p in strip_repeated_lines(extract_pages(pdf_path))]
    return pages_to_text(pages)
