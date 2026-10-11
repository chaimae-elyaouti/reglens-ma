import math

from reglens.evaluation import metrics as M
from reglens.generation.citations import (
    parse_llm_json,
    validate_article_citations,
)

G = {("d", "4")}


def test_ranking_metrics():
    ranked = [("d", "9"), ("d", "4"), ("d", "1")]
    assert M.hit_at_k(ranked, G, 1) == 0.0 and M.hit_at_k(ranked, G, 2) == 1.0
    assert M.reciprocal_rank(ranked, G) == 0.5
    assert M.recall_at_k(ranked, G, 3) == 1.0
    assert M.precision_at_k(ranked, G, 2) == 0.5
    assert M.reciprocal_rank([("d", "1")], G) == 0.0


def test_dedupe_keeps_first_position():
    assert M.dedupe([("d", "1"), ("d", "2"), ("d", "1")]) == [("d", "1"), ("d", "2")]


def test_percentile_and_citation_correctness():
    assert M.percentile([1, 2, 3, 4], 50) == 2.5
    assert math.isnan(M.percentile([], 50))
    assert M.citation_correctness([("d", "4"), ("d", "9")], G) == 0.5
    assert M.citation_correctness([], G) is None


def test_abstention_stats_and_sweep():
    s = M.abstention_stats([(True, True), (True, False), (False, True), (False, False)])
    assert s["accuracy"] == 0.5 and s["false_abstentions"] == 1 and s["missed_abstentions"] == 1
    sw = M.sweep_threshold([0.9, 0.8], [0.5, 0.6], [0.7])
    assert sw[0]["unanswerable_refused"] == 1.0 and sw[0]["answerable_wrongly_refused"] == 0.0


def test_parse_llm_json_variants():
    assert parse_llm_json('```json\n{"answer": "a", "citations": [1]}\n```')["answer"] == "a"
    assert parse_llm_json('Voici : {"answer": null, "citations": []} merci')["answer"] is None
    assert parse_llm_json("pas du json") is None


def test_validate_article_citations():
    answer = "Sans faute grave, un préavis est dû [Art. 43]. Voir aussi [Art. 99]."
    cleaned, cited, issues = validate_article_citations(
        answer, declared=["43", "99"], allowed={"13", "43", "61"}
    )

    assert cited == ["43"] and "[Art. 99]" not in cleaned
    assert any("hors_passages" in issue for issue in issues)


def test_validate_article_citations_uses_inline_and_requires_a_valid_citation():
    cleaned, cited, issues = validate_article_citations(
        "Le préavis est dû [Art 43].", declared=[], allowed={"43"}
    )
    assert cleaned == "Le préavis est dû [Art 43]."
    assert cited == ["43"] and issues == []

    _, cited, issues = validate_article_citations("Aucune citation.", [], {"43"})
    assert cited == [] and "reponse_sans_citation" in issues
