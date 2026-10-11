"""Retrieval hybride : dense (embeddings) + BM25 (mots-clés), fusionnés par RRF, filtres, rerank optionnel.

Pourquoi hybride pour du droit ? Les embeddings captent le sens ("licenciement" ~ "rupture du contrat"),
BM25 capte les termes exacts et les numéros ("article 39", "préavis"). Chacun rate ce que l'autre trouve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from rank_bm25 import BM25Okapi

from reglens.retrieval.fusion import rrf
from reglens.retrieval.store import VectorIndex
from reglens.retrieval.tokenize import tokenize_fr
from reglens.schemas import Chunk, Hit


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        # On indexe embed_text (avec l'en-tête "Article N") pour que la requête "article 39" matche.
        self.bm25 = BM25Okapi([tokenize_fr(c.embed_text) for c in chunks])

    def search(self, query: str, k: int, mask: np.ndarray | None) -> list[tuple[int, float]]:
        scores = np.asarray(self.bm25.get_scores(tokenize_fr(query)), dtype=np.float64)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        # BM25 renvoie des scores <= 0 pour les documents sans mot commun : on les écarte
        # (le notebook du tuto montrait "Orange juice" en 3e résultat pour "Cybertruck" à cause de ça).
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


def _topk(scores: np.ndarray, k: int) -> list[int]:
    idx = np.flatnonzero(np.isfinite(scores))
    if idx.size == 0:
        return []
    return idx[np.argsort(-scores[idx])][:k].tolist()


@dataclass
class RetrievalResult:
    hits: list[Hit]
    evidence_score: float | None  # meilleure similarité cosinus parmi les hits finaux
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, index: VectorIndex, embedder, *, candidates: int = 20, rrf_k: int = 60, reranker=None):
        self.index = index
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.bm25 = BM25Index(index.chunks)

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        mode: str = "hybrid",
        filters: dict[str, str] | None = None,
        rerank: bool = False,
    ) -> RetrievalResult:
        t0 = perf_counter()
        mask = self.index.mask_for(filters)
        qvec = self.embedder.embed_query(query)
        t_embed = perf_counter()

        sims = self.index.similarities(qvec)
        masked = np.where(mask, sims, -np.inf) if mask is not None else sims
        dense_rank = _topk(masked, self.candidates)
        bm25 = self.bm25.search(query, self.candidates, mask)
        bm25_rank = [i for i, _ in bm25]
        t_search = perf_counter()

        if mode == "dense":
            ranking = [(i, float(masked[i])) for i in dense_rank]
        elif mode == "bm25":
            ranking = bm25
        else:
            ranking = rrf([dense_rank, bm25_rank], k=self.rrf_k)

        rerank_scores: dict[int, float] = {}
        if rerank and self.reranker is not None and ranking:
            pool = [i for i, _ in ranking[: self.candidates]]
            scores = self.reranker.score(query, [self.index.chunks[i].text for i in pool])
            rerank_scores = dict(zip(pool, scores, strict=True))
            ranking = sorted(((i, rerank_scores[i]) for i in pool), key=lambda x: x[1], reverse=True)
        t_rerank = perf_counter()

        d_pos = {i: r for r, i in enumerate(dense_rank, 1)}
        b_pos = {i: r for r, i in enumerate(bm25_rank, 1)}
        hits = [
            Hit(
                chunk=self.index.chunks[i],
                score=float(s),
                dense_score=float(sims[i]),
                dense_rank=d_pos.get(i),
                bm25_rank=b_pos.get(i),
                rerank_score=rerank_scores.get(i),
            )
            for i, s in ranking[:k]
        ]
        evidence = max((h.dense_score for h in hits if h.dense_score is not None), default=None)
        return RetrievalResult(
            hits=hits,
            evidence_score=evidence,
            timings_ms={
                "embed_query": (t_embed - t0) * 1000,
                "search": (t_search - t_embed) * 1000,
                "rerank": (t_rerank - t_search) * 1000,
                "retrieval_total": (t_rerank - t0) * 1000,
            },
        )
