import numpy as np

from reglens.config import Settings
from reglens.ingestion.build_index import build_index_from_chunks
from reglens.ingestion.quality import quality_report
from reglens.retrieval.store import VectorIndex


def _build(chunks, embedder, out):
    return build_index_from_chunks(
        chunks, embedder, out, settings=Settings(_env_file=None),
        chunking_cfg={"strategy": "article"}, sources_meta=[{"doc_id": "d", "sha256": "abc", "as_of": "2026-01-01"}],
        expected_articles=12,
    )


def test_roundtrip_and_embedding_cache(chunks, embedder, tmp_path):
    m1 = _build(chunks, embedder, tmp_path)
    first_calls = embedder.calls
    assert first_calls == len(chunks) and m1["corpus_as_of"] == "2026-01-01"

    idx = VectorIndex.load(tmp_path)
    assert idx.embeddings.shape == (len(chunks), 128) and idx.chunks[0].id == chunks[0].id
    assert np.allclose(np.linalg.norm(idx.embeddings, axis=1), 1.0, atol=1e-5)

    m2 = _build(chunks, embedder, tmp_path)  # rien n'a changé : tout est réutilisé
    assert embedder.calls == first_calls and m2["index_version"] == m1["index_version"]


def test_version_changes_with_config(chunks, embedder, tmp_path):
    m1 = _build(chunks, embedder, tmp_path / "a")
    m2 = build_index_from_chunks(
        chunks, embedder, tmp_path / "b", settings=Settings(_env_file=None),
        chunking_cfg={"strategy": "article", "max_chars": 999},
        sources_meta=[{"doc_id": "d", "sha256": "abc", "as_of": ""}],
    )
    assert m1["index_version"] != m2["index_version"]


def test_quality_report_flags_problems(chunks):
    ok = quality_report(chunks, expected_articles=12)
    assert ok["n_articles"] == 12 and ok["missing_articles"] == [] and not ok["warnings"]
    bad = quality_report([c for c in chunks if c.article != "5"], expected_articles=12)
    assert 5 in bad["missing_articles"] and bad["warnings"]
