import importlib.util
import json
from pathlib import Path

import pytest

from reglens.ingestion.chunking import build_chunks_by_article
from reglens.ingestion.parse_pdf import load_document
from reglens.schemas import AskRequest  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def test_real_pdf_roundtrip(tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(6):  # 6 pages avec le même en-tête, pour tester la suppression d'en-têtes
        page = doc.new_page()
        art = i * 2 + 1
        page.insert_text(
            (50, 60),
            f"Bulletin officiel\nArticle {art}\nTexte de l'article {art} sur le salaire mensuel.\n"
            f"Article {art + 1}\nTexte de l'article {art + 1} sur le congé annuel payé.",
        )
    pdf = tmp_path / "t.pdf"
    doc.save(pdf)
    doc.close()

    text, offsets = load_document(pdf)
    assert "Bulletin officiel" not in text and len(offsets) == 6
    chunks = build_chunks_by_article(text, doc_id="t", doc_title="T", page_offsets=offsets)
    assert [c.article for c in chunks] == [str(n) for n in range(1, 13)]
    assert chunks[0].page == 1 and chunks[-1].page == 6


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOLDEN = [
    {"id": "a", "question": "congé annuel payé six mois", "answerable": True,
     "gold": [{"doc_id": "code-test", "article": "6"}]},
    {"id": "b", "question": "préavis de démission", "answerable": True,
     "gold": [{"doc_id": "code-test", "article": "7"}]},
    {"id": "c", "question": "taux de l'impôt sur le revenu", "answerable": False, "gold": []},
]


def test_eval_retrieval_metrics(retriever):
    ev = _load_run_eval()
    r = ev.eval_retrieval(retriever, GOLDEN, "hybrid", [1, 3])
    assert r["n_answerable"] == 2
    assert r["metrics"]["hit@3"] == 1.0 and r["metrics"]["mrr"] > 0.5
    assert r["evidence"]["unanswerable"] and r["latency_ms"]["p50"] >= 0


def test_eval_end_to_end_and_faithfulness(retriever, settings, tmp_path, capsys):
    import csv
    ev = _load_run_eval()
    import re

    from reglens.generation.llm import LLMResult
    from reglens.service import RagService

    class ContextLLM:
        def generate(self, messages):
            article = re.search(r"\[Art\.\s*(\d+)\]", messages[1]["content"]).group(1)
            text = json.dumps(
                {"answer": f"Réponse de test [Art. {article}].", "citations": [article]}
            )
            return LLMResult(text, "fake", "fake-1", 100, 20, 5.0)

    svc = RagService(retriever, ContextLLM(), settings)
    csv_path = tmp_path / "f.csv"
    out = ev.eval_end_to_end(svc, GOLDEN, "hybrid", 0.0, csv_path)
    assert out["abstention"]["missed_abstentions"] == 1  # le faux LLM répond aussi à la question hors corpus
    assert out["citation_correctness_mean"] is not None and csv_path.exists()
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert "congé annuel payé" in rows[0]["cited_passages"].lower()
    rows[0]["faithful"] = "1"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    ev.score_faithfulness(str(csv_path))
    assert "Faithfulness" in capsys.readouterr().out


def test_golden_validation(tmp_path):
    ev = _load_run_eval()
    bad = tmp_path / "g.jsonl"
    bad.write_text(json.dumps({"id": "x", "question": "q", "answerable": True}), encoding="utf-8")
    with pytest.raises(SystemExit):
        ev.load_golden(str(bad))
    ok = ev.load_golden(str(ROOT / "eval" / "golden_set.example.jsonl"))
    assert len(ok) == 3
