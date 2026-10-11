import json

from fastapi.testclient import TestClient

from reglens.api.app import create_app
from reglens.config import Settings
from reglens.schemas import AskRequest
from tests.conftest import make_service

Q = AskRequest(question="congé annuel payé après six mois")


def test_grounded_answer(retriever, settings, good_llm_text):
    svc, llm = make_service(retriever, settings, good_llm_text)
    r = svc.ask(Q, request_id="rid1")
    assert not r.abstain and r.cited == ["6"] and r.sources and r.request_id == "rid1"
    assert r.llm == "fake/fake-1" and r.tokens["prompt"] == 100 and "total" in r.timings_ms
    assert "avis juridique" in r.disclaimer and "2026-01-01" in r.disclaimer


def test_invalid_citation_only_means_abstain(retriever, settings):
    svc, _ = make_service(
        retriever, settings, json.dumps({"answer": "Texte [Art. 99].", "citations": ["99"]})
    )
    r = svc.ask(Q)
    assert r.abstain and r.abstain_reason == "ungrounded" and r.answer is None


def test_llm_abstention_and_unparseable(retriever, settings):
    svc, _ = make_service(retriever, settings, json.dumps({"answer": None, "citations": []}))
    assert svc.ask(Q).abstain_reason == "llm_abstained"
    svc, _ = make_service(retriever, settings, "n'importe quoi")
    assert svc.ask(Q).abstain_reason == "unparseable_output"


def test_low_evidence_skips_llm(retriever, good_llm_text):
    s = Settings(_env_file=None, min_evidence_score=0.99)
    svc, llm = make_service(retriever, s, good_llm_text)
    r = svc.ask(Q)
    assert r.abstain and r.abstain_reason == "low_evidence" and llm.calls == 0
    assert r.sources  # on montre quand même ce qui a été trouvé


def test_article_references_are_added_to_context(retriever, settings):
    from reglens.retrieval.retriever import RetrievalResult
    from reglens.schemas import Hit
    from reglens.service import RagService

    source_chunk = next(chunk for chunk in retriever.index.chunks if chunk.article == "5")
    source_chunk = source_chunk.model_copy(
        update={"text": "Article 5 renvoie aux articles 6 et 7 pour les conditions applicables."}
    )
    retriever.retrieve = lambda *args, **kwargs: RetrievalResult(
        hits=[Hit(chunk=source_chunk, score=1.0)], evidence_score=0.9
    )

    class CapturingLLM:
        def generate(self, messages):
            self.messages = messages
            return type("Result", (), {
                "text": json.dumps({"answer": "Voir aussi [Art. 6].", "citations": ["6"]}),
                "provider": "fake",
                "model": "fake-1",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            })()

    svc = RagService(retriever, CapturingLLM(), settings)
    response = svc.ask(AskRequest(question="Quelles sont les conditions applicables ?"))

    assert [source.article for source in response.sources] == ["5", "6", "7"]
    assert "article 6" in svc.llm.messages[1]["content"].lower()
    assert "article 7" in svc.llm.messages[1]["content"].lower()
    assert response.cited == ["6"]


def test_explicit_exception_reference_is_preserved_in_answer(retriever, settings):
    from reglens.retrieval.retriever import RetrievalResult
    from reglens.schemas import Hit
    from reglens.service import RagService

    source_chunk = next(chunk for chunk in retriever.index.chunks if chunk.article == "5")
    source_chunk = source_chunk.model_copy(
        update={
            "text": "La règle s'applique sous réserve des dérogations\nvisées aux articles 6 et 7."
        }
    )
    retriever.retrieve = lambda *args, **kwargs: RetrievalResult(
        hits=[Hit(chunk=source_chunk, score=1.0)], evidence_score=0.9
    )

    class OmittingLLM:
        def generate(self, messages):
            return type("Result", (), {
                "text": json.dumps({"answer": "Règle générale [Art. 5].", "citations": ["5"]}),
                "provider": "fake",
                "model": "fake-1",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            })()

    response = RagService(retriever, OmittingLLM(), settings).ask(
        AskRequest(question="Quelle est la règle applicable ?")
    )

    assert "sous réserve des dérogations" in response.answer
    assert "Article [Art. 6] : « Le congé annuel payé est accordé" in response.answer
    assert "Article [Art. 7] : « Le préavis de démission est prévu" in response.answer
    assert response.cited == ["5", "6", "7"]


def test_api_endpoints(retriever, settings, good_llm_text):
    svc, _ = make_service(retriever, settings, good_llm_text)
    client = TestClient(create_app(service=svc, settings=settings))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/version").json()["index_version"] == "testidx"

    ok = client.post("/ask", json={"question": "congé annuel payé après six mois"})
    assert ok.status_code == 200 and ok.json()["cited"] == ["6"] and ok.headers["x-request-id"]

    assert client.post("/ask", json={"question": "x"}).status_code == 422  # trop court (Pydantic)
    assert client.post("/ask", json={"question": "a" * 700}).status_code == 422  # trop long (métier)
    bad = client.post("/ask", json={"question": "congé annuel", "filters": {"inconnu": "x"}})
    assert bad.status_code == 422 and bad.json()["error"] == "invalid_request"

    assert client.get("/chunks/code-test:a6").json()["chunk"]["article"] == "6"
    assert client.get("/chunks/nope").status_code == 404


def test_api_llm_down_returns_503(retriever, settings):
    from reglens.errors import LLMUnavailable

    class Down:
        def generate(self, messages):
            raise LLMUnavailable("quota")

    svc, _ = make_service(retriever, settings, "")
    svc.llm = Down()
    r = TestClient(create_app(service=svc, settings=settings)).post("/ask", json={"question": "congé annuel payé"})
    assert r.status_code == 503 and r.json()["error"] == "llm_unavailable"


def test_rate_limit(retriever, good_llm_text):
    s = Settings(_env_file=None, rate_limit_per_ip_per_day=2)
    svc, _ = make_service(retriever, s, good_llm_text)
    client = TestClient(create_app(service=svc, settings=s))
    codes = [client.post("/ask", json={"question": "congé annuel payé"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
