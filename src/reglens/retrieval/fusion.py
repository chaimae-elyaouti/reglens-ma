"""Reciprocal Rank Fusion (même formule que le tuto, mais sur des identifiants, pas sur le texte)."""

from __future__ import annotations

from collections import defaultdict


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Chaque liste est un classement d'identifiants (meilleur en premier).
    score(id) = somme sur les listes de 1 / (k + rang). Renvoie [(id, score)] triés."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, ident in enumerate(ranking, 1):
            scores[ident] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
