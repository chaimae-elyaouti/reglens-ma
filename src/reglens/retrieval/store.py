"""Index vectoriel en lecture seule : une matrice numpy + les chunks.

Pourquoi pas Chroma ici ? Pour quelques milliers de chunks, un produit matriciel exact (numpy) prend
une fraction de milliseconde, n'a aucun serveur, aucun fichier SQLite à écrire (pratique en serverless,
où le disque est en lecture seule) et donne des résultats EXACTS (pas d'approximation HNSW).
Chroma/pgvector restent de bons candidats pour une expérience comparative (docs/EVALUATION.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reglens.errors import InvalidRequest
from reglens.schemas import Chunk

ALLOWED_FILTERS = {"doc_id", "lang", "article"}


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, manifest: dict):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Nombre de chunks différent du nombre de vecteurs : index corrompu.")
        self.chunks = chunks
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.manifest = manifest
        self.by_id = {c.id: i for i, c in enumerate(chunks)}

    @classmethod
    def load(cls, directory: str | Path) -> VectorIndex:
        d = Path(directory)
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(
                f"Index introuvable dans {d}. Lance d'abord : python -m reglens.ingestion.build_index"
            )
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
        return cls(chunks, np.load(d / "embeddings.npy"), manifest)

    def similarities(self, qvec: np.ndarray) -> np.ndarray:
        return self.embeddings @ np.asarray(qvec, dtype=np.float32)

    def mask_for(self, filters: dict[str, str] | None) -> np.ndarray | None:
        if not filters:
            return None
        bad = set(filters) - ALLOWED_FILTERS
        if bad:
            raise InvalidRequest(f"Filtre inconnu : {sorted(bad)}. Autorisés : {sorted(ALLOWED_FILTERS)}")
        return np.array(
            [all(getattr(c, f) == v for f, v in filters.items()) for c in self.chunks], dtype=bool
        )

    @property
    def version(self) -> str:
        return self.manifest.get("index_version", "")
