"""Modèles de données partagés (Pydantic) : ce qui circule entre modules et dans l'API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str  # stable et lisible : "code-travail:a39" ou "code-travail:a39-p2"
    doc_id: str
    doc_title: str
    article: str  # numéro d'article ("39"), "0" pour le préambule
    part: int = 1
    covers: list[str] = Field(default_factory=list)  # articles couverts (chunking à taille fixe)
    section_path: str = ""  # "LIVRE II > TITRE I ..." (meilleur effort)
    text: str  # texte affiché et envoyé au LLM
    embed_text: str  # texte réellement embeddé (avec en-tête de contexte)
    source_url: str = ""
    lang: str = "fr"
    as_of: str = ""
    page: int | None = None
    embed_hash: str = ""  # sha1(embed_text) : sert au cache d'embeddings

    def articles(self) -> list[str]:
        return self.covers or [self.article]


class Hit(BaseModel):
    chunk: Chunk
    score: float  # score utilisé pour l'ordre final (RRF, cosinus ou BM25 selon le mode)
    dense_score: float | None = None  # similarité cosinus (sert à l'abstention)
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


class Passage(BaseModel):
    """Un passage tel que montré à l'utilisateur (numéroté comme dans le prompt)."""

    n: int
    chunk_id: str
    doc_id: str
    doc_title: str
    article: str
    section_path: str
    page: int | None
    score: float
    dense_score: float | None
    text: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=5, ge=1, le=10)
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    filters: dict[str, str] | None = None
    rerank: bool = False


class AskResponse(BaseModel):
    answer: str | None
    abstain: bool
    abstain_reason: str | None = None
    cited: list[str] = Field(default_factory=list)  # numéros d'articles cités et validés
    sources: list[Passage] = Field(default_factory=list)  # tous les passages récupérés
    issues: list[str] = Field(default_factory=list)  # anomalies détectées (citations invalides...)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    tokens: dict[str, int | None] = Field(default_factory=dict)
    llm: str | None = None
    evidence_score: float | None = None
    index_version: str = ""
    corpus_as_of: str = ""
    disclaimer: str = ""
    request_id: str = ""
