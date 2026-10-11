"""Métriques d'évaluation, en fonctions pures (faciles à tester, aucune dépendance au modèle).

Vérité terrain au niveau ARTICLE : un résultat est correct si son article figure dans `gold`.
`ranked` est une liste ordonnée de (doc_id, article) SANS doublons (un article = une seule position).
"""

from __future__ import annotations

import math

Key = tuple[str, str]


def dedupe(ranked: list[Key]) -> list[Key]:
    seen: set[Key] = set()
    out = []
    for r in ranked:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def hit_at_k(ranked: list[Key], gold: set[Key], k: int) -> float:
    return 1.0 if any(r in gold for r in ranked[:k]) else 0.0


def recall_at_k(ranked: list[Key], gold: set[Key], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def precision_at_k(ranked: list[Key], gold: set[Key], k: int) -> float:
    return len(set(ranked[:k]) & gold) / k if k else 0.0


def reciprocal_rank(ranked: list[Key], gold: set[Key]) -> float:
    for i, r in enumerate(ranked, 1):
        if r in gold:
            return 1.0 / i
    return 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def citation_correctness(cited: list[Key], gold: set[Key]) -> float | None:
    """Part des articles cités qui font partie des bons articles. None si rien n'est cité."""
    if not cited:
        return None
    return sum(1 for c in set(cited) if c in gold) / len(set(cited))


def abstention_stats(rows: list[tuple[bool, bool]]) -> dict:
    """rows = [(abstained, should_abstain)]. 'Positif' = s'abstenir."""
    tp = sum(1 for a, s in rows if a and s)
    fp = sum(1 for a, s in rows if a and not s)  # a refusé alors qu'une réponse existait
    fn = sum(1 for a, s in rows if not a and s)  # a répondu alors qu'il fallait refuser
    n = len(rows)
    return {
        "accuracy": (sum(1 for a, s in rows if a == s) / n) if n else math.nan,
        "abstain_precision": tp / (tp + fp) if tp + fp else math.nan,
        "abstain_recall": tp / (tp + fn) if tp + fn else math.nan,
        "false_abstentions": fp,
        "missed_abstentions": fn,
    }


def sweep_threshold(
    answerable_scores: list[float], unanswerable_scores: list[float], thresholds: list[float]
) -> list[dict]:
    """Pour chaque seuil t (on s'abstient si score < t) : quelle part des questions SANS réponse est refusée,
    et quelle part des questions AVEC réponse est refusée à tort ? À toi de choisir le compromis."""
    out = []
    for t in thresholds:
        out.append(
            {
                "threshold": round(t, 4),
                "unanswerable_refused": (
                    sum(1 for s in unanswerable_scores if s < t) / len(unanswerable_scores)
                    if unanswerable_scores else math.nan
                ),
                "answerable_wrongly_refused": (
                    sum(1 for s in answerable_scores if s < t) / len(answerable_scores)
                    if answerable_scores else math.nan
                ),
            }
        )
    return out
