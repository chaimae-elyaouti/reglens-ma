import hashlib
import json

import numpy as np
import pytest

from reglens.config import Settings
from reglens.generation.llm import LLMResult
from reglens.ingestion.chunking import build_chunks_by_article
from reglens.retrieval.retriever import Retriever
from reglens.retrieval.store import VectorIndex
from reglens.retrieval.tokenize import tokenize_fr
from reglens.service import RagService

DIM = 128


class HashEmbedder:
    """Faux embedder déterministe (sac de mots hashés) : rapide, sans réseau, suffisant pour tester la plomberie."""

    dim = DIM
    max_seq_length = None

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = np.zeros(DIM, dtype=np.float32)
        for t in tokenize_fr(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def generate(self, messages):
        self.calls += 1
        return LLMResult(self.text, "fake", "fake-1", 100, 20, 5.0)


CORPUS = """LIVRE I
Dispositions générales
TITRE I : CHAMP D'APPLICATION
Article premier
Le présent code s'applique aux salariés et aux employeurs du secteur privé.
Article 2
Est salarié toute personne engagée par un employeur pour un travail contre rémunération.
Article 3
Le salaire est versé en monnaie ayant cours légal. Le paiement du salaire est mensuel.
TITRE II : CONTRAT DE TRAVAIL
Article 4
Le licenciement du salarié sans motif valable ouvre droit à une indemnité de licenciement.
Article 5
La durée normale du travail est fixée par semaine. Les heures supplémentaires sont majorées.
Article 6
Le congé annuel payé est accordé au salarié après six mois de service continu.
Article 7
Le préavis de démission est prévu par le contrat ou la convention collective.
Article 8
Le salarié victime d'un accident du travail bénéficie d'une réparation.
Article 9
Le harcèlement sexuel au travail est interdit et sanctionné.
Article 10
Les délégués du personnel sont élus dans les établissements d'au moins dix salariés.
Article 11
La grève est un droit reconnu aux salariés.
Article 12
Le travail de nuit des femmes est réglementé.
"""


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def chunks():
    return build_chunks_by_article(CORPUS, doc_id="code-test", doc_title="Code du travail (test)", as_of="2026-01-01")


@pytest.fixture
def index(chunks, embedder):
    vecs = embedder.embed_passages([c.embed_text for c in chunks])
    return VectorIndex(chunks, vecs, {"index_version": "testidx", "corpus_as_of": "2026-01-01", "n_chunks": len(chunks)})


@pytest.fixture
def retriever(index, embedder):
    return Retriever(index, embedder, candidates=10, rrf_k=60)


@pytest.fixture
def settings():
    return Settings(_env_file=None, min_evidence_score=0.0, rate_limit_per_ip_per_day=0)


def make_service(retriever, settings, llm_text):
    llm = FakeLLM(llm_text)
    return RagService(retriever, llm, settings), llm


@pytest.fixture
def good_llm_text():
    return json.dumps({"answer": "Le congé annuel est accordé après six mois [Art. 6].", "citations": ["6"]})
