"""Construction de l'index : PDF -> chunks -> embeddings -> dossier versionné.

    python -m reglens.ingestion.build_index                      # chunking par article (défaut)
    python -m reglens.ingestion.build_index --strategy fixed --size 1000 --out index/fixed1000

Reproductibilité : le manifest enregistre le hash du corpus, la config de chunking, le modèle d'embeddings
et en déduit `index_version`. Si rien n'a changé, la version est identique ; sinon elle change.
Les embeddings déjà calculés (même texte + même modèle) sont réutilisés : relancer coûte quelques secondes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reglens.config import Settings, get_settings
from reglens.ingestion.chunking import build_chunks_by_article, build_chunks_fixed
from reglens.ingestion.parse_pdf import load_document
from reglens.ingestion.quality import quality_report
from reglens.logging_setup import log_event, setup_logging
from reglens.schemas import Chunk


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def embed_with_cache(chunks: list[Chunk], embedder, previous: tuple[list[Chunk], np.ndarray] | None):
    """Ne recalcule que les chunks dont (embed_text) n'a jamais été embeddé avec ce modèle."""
    cache: dict[str, np.ndarray] = {}
    if previous is not None:
        old_chunks, old_vecs = previous
        cache = {c.embed_hash: old_vecs[i] for i, c in enumerate(old_chunks)}
    todo = [i for i, c in enumerate(chunks) if c.embed_hash not in cache]
    dim = embedder.dim
    vecs = np.zeros((len(chunks), dim), dtype=np.float32)
    for i, c in enumerate(chunks):
        if c.embed_hash in cache:
            vecs[i] = cache[c.embed_hash]
    if todo:
        new = embedder.embed_passages([chunks[i].embed_text for i in todo])
        for j, i in enumerate(todo):
            vecs[i] = new[j]
    log_event("embeddings_done", computed=len(todo), reused=len(chunks) - len(todo))
    return vecs


def write_index(out_dir: Path, chunks: list[Chunk], vecs: np.ndarray, manifest: dict, quality: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    np.save(out_dir / "embeddings.npy", vecs.astype(np.float32))
    (out_dir / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_index_from_chunks(
    chunks: list[Chunk],
    embedder,
    out_dir: str | Path,
    *,
    settings: Settings,
    chunking_cfg: dict,
    sources_meta: list[dict],
    expected_articles: int | None = None,
) -> dict:
    out = Path(out_dir)
    previous = None
    if (out / "manifest.json").exists():
        old = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if old.get("embedding_model") == settings.embedding_model and old.get("passage_prefix") == settings.passage_prefix:
            with open(out / "chunks.jsonl", encoding="utf-8") as f:
                previous = ([Chunk.model_validate_json(ln) for ln in f if ln.strip()], np.load(out / "embeddings.npy"))

    quality = quality_report(chunks, expected_articles)
    for w in quality["warnings"]:
        log_event("quality_warning", level=logging.WARNING, warning=w)

    truncated = None
    if hasattr(embedder, "count_tokens") and getattr(embedder, "max_seq_length", None):
        counts = embedder.count_tokens([c.embed_text for c in chunks])
        truncated = sum(1 for n in counts if n > embedder.max_seq_length)
        if truncated:
            log_event(
                "chunks_truncated_by_model", level=logging.WARNING, n=truncated,
                max_seq_length=embedder.max_seq_length,
                hint="réduis max_chars ou choisis un modèle à contexte plus long",
            )
    quality["chunks_truncated_by_model"] = truncated

    vecs = embed_with_cache(chunks, embedder, previous)
    corpus_hash = hashlib.sha256("|".join(sorted(s["sha256"] for s in sources_meta)).encode()).hexdigest()
    cfg = {
        "corpus_hash": corpus_hash, "chunking": chunking_cfg,
        "embedding_model": settings.embedding_model, "passage_prefix": settings.passage_prefix,
    }
    as_of_values = [s["as_of"] for s in sources_meta if s.get("as_of")]
    manifest = {
        "index_version": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10],
        **cfg,
        "query_prefix": settings.query_prefix,
        "dim": int(vecs.shape[1]) if len(vecs) else 0,
        "n_chunks": len(chunks),
        "sources": sources_meta,
        "corpus_as_of": max(as_of_values) if as_of_values else "",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_index(out, chunks, vecs, manifest, quality)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="data/sources.json")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default=None, help="défaut : INDEX_DIR du .env")
    ap.add_argument("--strategy", choices=["article", "fixed"], default="article")
    ap.add_argument("--max-chars", type=int, default=1500, help="article : taille max avant découpe")
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--size", type=int, default=1000, help="fixed : taille de fenêtre")
    ap.add_argument("--no-header", action="store_true", help="article : pas d'en-tête de contexte")
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    from reglens.retrieval.embedder import SentenceTransformerEmbedder

    docs = [d for d in json.loads(Path(args.sources).read_text(encoding="utf-8"))["documents"] if d.get("enabled")]
    all_chunks: list[Chunk] = []
    sources_meta: list[dict] = []
    expected = 0
    for d in docs:
        pdf = Path(args.raw) / d["local_file"]
        if not pdf.exists():
            raise SystemExit(f"PDF manquant : {pdf}. Lance : python scripts/fetch_corpus.py")
        text, offsets = load_document(pdf)
        common = dict(
            doc_id=d["doc_id"], doc_title=d["title"], source_url=d.get("url", ""), lang=d.get("lang", "fr"),
            as_of=d.get("as_of", ""), page_offsets=offsets,
        )
        if args.strategy == "article":
            chunks = build_chunks_by_article(
                text, max_chars=args.max_chars, overlap=args.overlap, with_header=not args.no_header, **common
            )
            cfg = {"strategy": "article", "max_chars": args.max_chars, "overlap": args.overlap,
                   "header": not args.no_header}
        else:
            chunks = build_chunks_fixed(text, size=args.size, overlap=args.overlap, **common)
            cfg = {"strategy": "fixed", "size": args.size, "overlap": args.overlap}
        log_event("document_chunked", doc_id=d["doc_id"], n_chunks=len(chunks), chars=len(text))
        all_chunks += chunks
        expected += d.get("expected_articles") or 0
        sources_meta.append({"doc_id": d["doc_id"], "url": d.get("url", ""), "as_of": d.get("as_of", ""),
                             "sha256": _sha256_file(pdf)})

    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, settings.query_prefix, settings.passage_prefix,
        settings.embedding_device, settings.embedding_batch_size,
    )
    manifest = build_index_from_chunks(
        all_chunks, embedder, args.out or settings.index_dir, settings=settings, chunking_cfg=cfg,
        sources_meta=sources_meta, expected_articles=expected or None,
    )
    print(json.dumps({k: manifest[k] for k in ("index_version", "n_chunks", "dim", "corpus_as_of")}, indent=2))
    print(f"Rapport qualité : {Path(args.out or settings.index_dir) / 'quality.json'}")


if __name__ == "__main__":
    main()
