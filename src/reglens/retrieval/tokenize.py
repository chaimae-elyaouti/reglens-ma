"""Tokenizer français pour BM25.

Le tokenizer par défaut de BM25Retriever (LangChain) coupe sur les espaces : "l'employeur" et "employeur"
deviennent deux mots différents, "Congés" != "conges". Ici : minuscules, accents retirés, élisions
(l', d', qu'...) enlevées, mots vides courants supprimés.
"""

from __future__ import annotations

import re
import unicodedata

_ELISION = re.compile(r"\b(?:[ldjmtsc]|qu|jusqu|lorsqu|puisqu)['’]")
STOPWORDS = frozenset(
    "le la les l un une des du de d et ou a au aux en dans pour par sur avec ce ces se sa son ses "
    "qui que quoi est sont etre ont il elle ils elles ne pas plus ainsi lors si sous entre".split()
)


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize_fr(text: str) -> list[str]:
    text = strip_accents(text.lower())
    text = _ELISION.sub(" ", text)
    return [t for t in re.findall(r"\w+", text) if t not in STOPWORDS]
