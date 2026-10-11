"""Reranking optionnel par cross-encoder (local).

Un cross-encoder lit (question, passage) ENSEMBLE : plus précis que la similarité d'embeddings, mais il
faut un appel par candidat => latence CPU. On ne reranke donc que les ~20 meilleurs candidats, et c'est
une EXPÉRIENCE à mesurer (gain de Hit@K vs latence ajoutée), pas une évidence.
"""

from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]
