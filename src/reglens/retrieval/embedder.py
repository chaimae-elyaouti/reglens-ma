"""Embeddings locaux (gratuits, CPU).

Point d'attention RAM/CPU : le modèle est chargé UNE fois (plusieurs centaines de Mo en mémoire avec PyTorch).
`max_seq_length` : tout texte plus long est TRONQUÉ sans erreur. build_index compare cette limite à la
taille de tes chunks et t'avertit.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    max_seq_length: int | None

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        device: str = "cpu",
        batch_size: int = 16,
    ):
        from sentence_transformers import SentenceTransformer  # import tardif : lourd (PyTorch)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.max_seq_length = getattr(self.model, "max_seq_length", None)
        self.dim = int(self._encode(["test"]).shape[1])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # vecteurs unitaires => produit scalaire = cosinus
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    def count_tokens(self, texts: list[str]) -> list[int]:
        tok = self.model.tokenizer
        return [len(tok(self.passage_prefix + t)["input_ids"]) for t in texts]
