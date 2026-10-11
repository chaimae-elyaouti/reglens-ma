import pytest

from reglens.errors import InvalidRequest
from reglens.retrieval.fusion import rrf
from reglens.retrieval.tokenize import tokenize_fr


def test_tokenizer_french():
    toks = tokenize_fr("L'employeur doit-il verser d'un salaire ? Les congés")
    assert "employeur" in toks and "salaire" in toks and "conges" in toks
    assert "les" not in toks and "d" not in toks


def test_rrf_prefers_items_ranked_by_both_lists():
    fused = rrf([[1, 2, 3], [3, 2, 4]], k=60)
    assert [i for i, _ in fused][:3] == [3, 2, 1]
    assert {i for i, _ in fused} == {1, 2, 3, 4}


def test_dense_finds_semantic_neighbor(retriever):
    res = retriever.retrieve("congé annuel payé après six mois de service", k=3, mode="dense")
    assert res.hits[0].chunk.article == "6"
    assert res.evidence_score is not None and res.evidence_score > 0


def test_bm25_finds_exact_article_number(retriever):
    res = retriever.retrieve("article 9", k=3, mode="bm25")
    assert res.hits and res.hits[0].chunk.article == "9"


def test_bm25_returns_nothing_without_common_terms(retriever):
    assert retriever.retrieve("zzzqqq xxxyyy", k=3, mode="bm25").hits == []


def test_hybrid_contains_both_signals(retriever):
    res = retriever.retrieve("licenciement indemnité", k=5, mode="hybrid")
    assert res.hits[0].chunk.article == "4"
    assert res.hits[0].bm25_rank is not None and res.hits[0].dense_rank is not None
    assert set(res.timings_ms) >= {"embed_query", "search", "retrieval_total"}


def test_filters(retriever):
    assert retriever.retrieve("salaire", k=5, filters={"doc_id": "autre"}).hits == []
    assert retriever.retrieve("salaire", k=5, filters={"doc_id": "code-test"}).hits
    with pytest.raises(InvalidRequest):
        retriever.retrieve("salaire", filters={"champ_inconnu": "x"})
