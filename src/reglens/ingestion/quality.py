"""Contrôle qualité de l'ingestion : on ne fait jamais confiance à un parsing sans le mesurer."""

from __future__ import annotations

import statistics

from reglens.schemas import Chunk


def quality_report(chunks: list[Chunk], expected_articles: int | None = None) -> dict:
    nums = sorted({int(c.article) for c in chunks if c.article.isdigit() and int(c.article) > 0})
    present = set(nums)
    missing = [n for n in range(1, nums[-1] + 1) if n not in present] if nums else []
    lengths = [len(c.text) for c in chunks] or [0]
    tiny = [c.id for c in chunks if len(c.text) < 40]
    ids = [c.id for c in chunks]
    warnings: list[str] = []
    if not chunks:
        warnings.append("Aucun chunk : le PDF est-il un scan ? l'expression 'Article N' est-elle présente ?")
    if expected_articles and len(nums) < 0.95 * expected_articles:
        warnings.append(f"{len(nums)} articles trouvés pour ~{expected_articles} attendus.")
    if missing:
        warnings.append(f"{len(missing)} numéros manquants (ex: {missing[:10]}).")
    if tiny:
        warnings.append(f"{len(tiny)} chunks < 40 caractères (ex: {tiny[:5]}) : en-têtes ou renvois ?")
    if len(ids) != len(set(ids)):
        warnings.append("Identifiants de chunk dupliqués.")
    return {
        "n_chunks": len(chunks),
        "n_articles": len(nums),
        "max_article": nums[-1] if nums else None,
        "missing_articles": missing,
        "chars_min": min(lengths),
        "chars_median": int(statistics.median(lengths)),
        "chars_max": max(lengths),
        "tiny_chunks": tiny,
        "warnings": warnings,
    }
