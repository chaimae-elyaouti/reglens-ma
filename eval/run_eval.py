"""Évaluation reproductible du RAG.

Retrieval seul (aucune clé API, rapide) :
    python eval/run_eval.py --golden eval/golden_set.jsonl --index index/v1 --modes dense,bm25,hybrid

Comparer plusieurs chunkings (un index par stratégie) :
    python eval/run_eval.py --index index/v1 index/fixed1000 --modes hybrid

Calibrer le seuil d'abstention :
    python eval/run_eval.py --sweep

De bout en bout (génération, abstention, citations) — consomme ton quota LLM, résultats mis en cache :
    python eval/run_eval.py --with-llm --sleep 2.5
    # puis annote la colonne `faithful` (0/1) du CSV généré, et :
    python eval/run_eval.py --score-faithfulness eval/faithfulness_XXXX.csv

Rien n'est inventé : tous les chiffres sortent de TON jeu de questions, avec TON index.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.config import get_settings  # noqa: E402
from reglens.evaluation import metrics as M  # noqa: E402
from reglens.retrieval.embedder import SentenceTransformerEmbedder  # noqa: E402
from reglens.retrieval.retriever import Retriever  # noqa: E402
from reglens.retrieval.store import VectorIndex  # noqa: E402
from reglens.schemas import AskRequest  # noqa: E402


def load_golden(path: str) -> list[dict]:
    rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    for r in rows:
        for key in ("id", "question", "answerable"):
            if key not in r:
                raise SystemExit(f"Question invalide (champ '{key}' manquant) : {r}")
        if r["answerable"] and not r.get("gold"):
            raise SystemExit(f"{r['id']} : answerable=true exige 'gold' (doc_id + article).")
    return rows


def gold_of(q: dict) -> set[M.Key]:
    return {(g["doc_id"], str(g["article"])) for g in q.get("gold", [])}


def ranked_keys(hits) -> list[M.Key]:
    return M.dedupe([(h.chunk.doc_id, a) for h in hits for a in h.chunk.articles()])


def fmt(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{x:.3f}"


def eval_retrieval(retriever: Retriever, golden: list[dict], mode: str, ks: list[int]) -> dict:
    kmax = max(ks)
    per_q, lat = [], []
    agg = {f"hit@{k}": [] for k in ks} | {f"recall@{k}": [] for k in ks} | {f"precision@{k}": [] for k in ks}
    agg["mrr"] = []
    ev_ans, ev_unans = [], []
    for q in golden:
        res = retriever.retrieve(q["question"], k=kmax, mode=mode)
        lat.append(res.timings_ms["retrieval_total"])
        if res.evidence_score is not None:
            (ev_ans if q["answerable"] else ev_unans).append(res.evidence_score)
        if not q["answerable"]:
            continue
        ranked, gold = ranked_keys(res.hits), gold_of(q)
        for k in ks:
            agg[f"hit@{k}"].append(M.hit_at_k(ranked, gold, k))
            agg[f"recall@{k}"].append(M.recall_at_k(ranked, gold, k))
            agg[f"precision@{k}"].append(M.precision_at_k(ranked, gold, k))
        agg["mrr"].append(M.reciprocal_rank(ranked, gold))
        per_q.append({"id": q["id"], "rr": agg["mrr"][-1], "top": [list(r) for r in ranked[:5]],
                      "gold": [list(g) for g in gold]})
    return {
        "n_answerable": len(per_q),
        "metrics": {k: statistics.fmean(v) if v else None for k, v in agg.items()},
        "latency_ms": {"p50": M.percentile(lat, 50), "p95": M.percentile(lat, 95)},
        "evidence": {"answerable": ev_ans, "unanswerable": ev_unans},
        "per_question": per_q,
    }


def eval_end_to_end(svc, golden: list[dict], mode: str, sleep: float, csv_path: Path) -> dict:
    rows, lat, abst, cites, toks = [], [], [], [], []
    for q in golden:
        resp = svc.ask(AskRequest(question=q["question"], mode=mode))
        lat.append(resp.timings_ms.get("total", 0.0))
        abst.append((resp.abstain, not q["answerable"]))
        if q["answerable"] and not resp.abstain:
            cited = [(p.doc_id, p.article) for p in resp.sources if p.article in resp.cited]
            c = M.citation_correctness(cited, gold_of(q))
            if c is not None:
                cites.append(c)
        if resp.tokens.get("prompt"):
            toks.append((resp.tokens["prompt"] or 0) + (resp.tokens.get("completion") or 0))
        rows.append({"id": q["id"], "question": q["question"], "abstained": resp.abstain,
                     "answer": resp.answer or "",
                     "cited_passages": " || ".join(
                         p.text[:300] for p in resp.sources if p.article in resp.cited
                     ),
                     "faithful": "", "notes": ""})
        time.sleep(sleep)  # respecte les limites de débit du plan gratuit
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {
        "abstention": M.abstention_stats(abst),
        "citation_correctness_mean": statistics.fmean(cites) if cites else None,
        "latency_total_ms": {"p50": M.percentile(lat, 50), "p95": M.percentile(lat, 95)},
        "tokens_per_query_mean": statistics.fmean(toks) if toks else None,
        "faithfulness_csv": str(csv_path),
    }


def score_faithfulness(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        vals = [r["faithful"].strip() for r in csv.DictReader(f) if r["faithful"].strip() in ("0", "1")]
    if not vals:
        raise SystemExit("Aucune annotation 0/1 trouvée dans la colonne 'faithful'.")
    print(f"Faithfulness (annotation manuelle) : {sum(map(int, vals))}/{len(vals)} = "
          f"{sum(map(int, vals)) / len(vals):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--index", nargs="+", default=None)
    ap.add_argument("--modes", default="dense,bm25,hybrid")
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.5)
    ap.add_argument("--score-faithfulness", default=None)
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()

    if args.score_faithfulness:
        return score_faithfulness(args.score_faithfulness)

    settings = get_settings()
    golden = load_golden(args.golden)
    ks = [int(k) for k in args.ks.split(",")]
    modes = args.modes.split(",")
    indexes = args.index or [settings.index_dir]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, settings.query_prefix, settings.passage_prefix,
        settings.embedding_device, settings.embedding_batch_size,
    )
    report = {"date": stamp, "golden": args.golden, "n_questions": len(golden),
              "embedding_model": settings.embedding_model, "runs": []}
    table = ["| index | mode | n | " + " | ".join(f"Hit@{k}" for k in ks) + f" | Recall@{ks[-1]} | MRR | p50 ms | p95 ms |",
             "|---|---|---|" + "---|" * (len(ks) + 4)]
    for idx_dir in indexes:
        index = VectorIndex.load(idx_dir)
        retriever = Retriever(index, embedder, candidates=settings.candidates, rrf_k=settings.rrf_k)
        for mode in modes:
            r = eval_retrieval(retriever, golden, mode, ks)
            r.update({"index": idx_dir, "index_version": index.version, "mode": mode})
            report["runs"].append(r)
            m = r["metrics"]
            table.append(f"| {idx_dir} | {mode} | {r['n_answerable']} | "
                         + " | ".join(fmt(m[f"hit@{k}"]) for k in ks)
                         + f" | {fmt(m[f'recall@{ks[-1]}'])} | {fmt(m['mrr'])} | "
                         f"{r['latency_ms']['p50']:.0f} | {r['latency_ms']['p95']:.0f} |")
    print("\n".join(table))

    if args.sweep:
        run = next((r for r in report["runs"] if r["mode"] in ("hybrid", "dense")), report["runs"][0])
        a, u = run["evidence"]["answerable"], run["evidence"]["unanswerable"]
        if not a or not u:
            print("\nSweep impossible : il faut des questions answerable=true ET answerable=false.")
        else:
            lo, hi = min(a + u), max(a + u)
            ths = [lo + (hi - lo) * i / 12 for i in range(13)]
            print("\nSeuil d'abstention (cosinus) :")
            print("| seuil | questions sans réponse refusées | questions avec réponse refusées à tort |\n|---|---|---|")
            for row in M.sweep_threshold(a, u, ths):
                print(f"| {row['threshold']} | {fmt(row['unanswerable_refused'])} | "
                      f"{fmt(row['answerable_wrongly_refused'])} |")

    if args.with_llm:
        from reglens.service import build_service

        svc = build_service(settings, cache_llm=True)
        mode = "hybrid" if "hybrid" in modes else modes[0]
        report["end_to_end"] = eval_end_to_end(svc, golden, mode, args.sleep,
                                               Path(f"eval/faithfulness_{stamp}.csv"))
        print("\nBout en bout :", json.dumps(report["end_to_end"], ensure_ascii=False, indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "latest.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"\nRésultats : {out / f'{stamp}.json'}")


if __name__ == "__main__":
    main()
