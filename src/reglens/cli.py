"""Interface en ligne de commande.

    python -m reglens.cli ask "Quelle est la durée normale du travail ?" --no-llm
    python -m reglens.cli ask "..." --mode dense --k 8

--no-llm : n'affiche que les passages retrouvés (aucune clé API nécessaire). C'est ton outil n°1
pour juger le retrieval à l'œil avant de brancher le LLM.
"""

from __future__ import annotations

import argparse

from reglens.config import get_settings
from reglens.logging_setup import setup_logging
from reglens.schemas import AskRequest
from reglens.service import build_service


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    a.add_argument("--doc", default=None, help="filtre doc_id")
    a.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    setup_logging("WARNING")
    svc = build_service(settings, with_llm=not args.no_llm)
    filters = {"doc_id": args.doc} if args.doc else None

    if args.no_llm:
        res = svc.retriever.retrieve(args.question, k=args.k, mode=args.mode, filters=filters)
        print(f"evidence (cosinus max) = {res.evidence_score}")
        for i, h in enumerate(res.hits, 1):
            print(f"\n[{i}] {h.chunk.doc_id} art.{h.chunk.article} p.{h.chunk.page} "
                  f"score={h.score:.4f} dense={h.dense_score:.3f} d#{h.dense_rank} b#{h.bm25_rank}")
            print(h.chunk.text[:400])
        print({k: round(v, 1) for k, v in res.timings_ms.items()})
        return

    resp = svc.ask(AskRequest(question=args.question, k=args.k, mode=args.mode, filters=filters))
    print(resp.answer if not resp.abstain else f"[Abstention : {resp.abstain_reason}]")
    for p in resp.sources:
        mark = "*" if p.article in resp.cited else " "
        print(f" {mark}[Art. {p.article}] {p.doc_id} p.{p.page} (cosinus {p.dense_score})")
    print(resp.timings_ms, resp.tokens, resp.issues)


if __name__ == "__main__":
    main()
