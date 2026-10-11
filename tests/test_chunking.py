from reglens.ingestion.chunking import (
    build_chunks_by_article,
    build_chunks_fixed,
    find_article_starts,
    split_long,
)
from reglens.ingestion.parse_pdf import clean_text, pages_to_text, strip_repeated_lines
from tests.conftest import CORPUS


def test_one_chunk_per_article_with_stable_ids(chunks):
    assert [c.id for c in chunks] == [f"code-test:a{n}" for n in range(1, 13)]
    assert chunks[0].article == "1"  # "Article premier" -> 1


def test_section_path_and_no_heading_leak(chunks):
    assert chunks[0].section_path == "LIVRE I > TITRE I : CHAMP D'APPLICATION"
    assert chunks[3].section_path == "LIVRE I > TITRE II : CONTRAT DE TRAVAIL"
    assert "TITRE II" not in chunks[2].text  # l'en-tête suivant n'est pas aspiré par l'article précédent


def test_embed_text_has_context_header(chunks):
    assert chunks[5].embed_text.startswith("Code du travail (test) — Article 6")


def test_glued_label_is_fixed():
    text = "Article 1Le texte un.\nArticle 2Le texte deux."
    chunks = build_chunks_by_article(text, doc_id="d", doc_title="D")
    assert chunks[0].text.startswith("Article 1 Le texte")
    assert len(chunks) == 2


def test_cross_reference_at_sentence_start_is_not_an_article():
    text = "Article 1 a. Article 2 b.\nArticle 4 renvoi qui ressemble à un début.\nArticle 3 c.\nArticle 4 d."
    assert [n for _, n in find_article_starts(text)] == [1, 2, 3, 4]


def test_lowercase_reference_ignored():
    text = "Article 1 voir l'article 2 ci-dessous.\nArticle 2 suite."
    assert [n for _, n in find_article_starts(text)] == [1, 2]


def test_gap_in_numbering_is_tolerated():
    text = "Article 1 a.\nArticle 2 b.\nArticle 5 c."
    assert [n for _, n in find_article_starts(text)] == [1, 2, 5]


def test_long_article_split_with_suffix_ids_and_size_limit():
    sentence = "Le salarié a droit à une indemnité calculée selon l'ancienneté. "
    text = "Article 1 " + sentence * 60 + "\nArticle 2 court."
    chunks = build_chunks_by_article(text, doc_id="d", doc_title="D", max_chars=500, overlap=60)
    ids = [c.id for c in chunks]
    assert ids[0] == "d:a1" and ids[1] == "d:a1-p2" and ids[-1] == "d:a2"
    assert all(len(c.text) <= 500 for c in chunks)


def test_split_long_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        split_long("x" * 100, 50, 50)


def test_fixed_chunking_labels_and_covers():
    chunks = build_chunks_fixed(CORPUS, doc_id="d", doc_title="D", size=300, overlap=30)
    assert len(chunks) > 3
    assert all(c.articles() for c in chunks)
    assert any(len(c.covers) > 1 for c in chunks)  # une fenêtre recouvre plusieurs articles


def test_pages_offsets_and_page_number():
    text, offsets = pages_to_text(["Article 1 a.", "Article 2 b."])
    chunks = build_chunks_by_article(text, doc_id="d", doc_title="D", page_offsets=offsets)
    assert [c.page for c in chunks] == [1, 2]


def test_strip_repeated_headers_and_clean():
    pages = [f"Bulletin officiel\ncorps {i}\n{i}" for i in range(6)]
    cleaned = strip_repeated_lines(pages)
    assert all("Bulletin officiel" not in p for p in cleaned)
    assert clean_text("licen-\nciement   du\u00a0salarié") == "licenciement du salarié"
