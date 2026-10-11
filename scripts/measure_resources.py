"""Mesure le service RAG local sans appeler un fournisseur LLM."""

from __future__ import annotations

import os
import statistics
import time

import psutil

MIN_AVAILABLE_GB = 1.5


def main() -> None:
    memory = psutil.virtual_memory()
    if memory.available < MIN_AVAILABLE_GB * 1_000_000_000:
        available_gb = memory.available / 1_000_000_000
        raise SystemExit(
            f"RAM disponible insuffisante ({available_gb:.2f} Go). "
            f"Libère au moins {MIN_AVAILABLE_GB:.1f} Go, puis relance le script."
        )

    start = time.perf_counter()
    from reglens.config import get_settings
    from reglens.service import build_service

    settings = get_settings()
    settings.use_reranker = False
    service = build_service(settings, with_llm=False)
    load_seconds = time.perf_counter() - start
    process = psutil.Process(os.getpid())
    idle_rss_mb = process.memory_info().rss / 1_000_000

    questions = [
        "Quelle est la durée normale du travail ?",
        "Mon employeur peut-il me licencier sans préavis ?",
        "Combien de jours de congé annuel ai-je ?",
        "Quels sont les droits d'une salariée enceinte ?",
        "Que dit le Code sur le salaire minimum ?",
        "Qu'est-ce qu'une faute grave ?",
        "Comment se calcule l'indemnité de licenciement ?",
        "Un employeur peut-il baisser mon salaire ?",
        "Quelle est la durée de la période d'essai ?",
        "Les heures supplémentaires sont-elles majorées ?",
    ]

    latencies_ms = []
    for question in questions:
        query_start = time.perf_counter()
        service.retriever.retrieve(question, k=5, rerank=False)
        latencies_ms.append((time.perf_counter() - query_start) * 1000)

    memory_info = process.memory_info()
    peak_memory_mb = getattr(memory_info, "peak_wset", memory_info.rss) / 1_000_000
    print(f"Démarrage (chargement) : {load_seconds:.1f} s")
    print(f"RAM au repos du service: {idle_rss_mb:.0f} Mo")
    print(f"Pic RAM du processus  : {peak_memory_mb:.0f} Mo")
    print("LLM et reranker       : désactivés")
    print(f"Latence retrieval     : médiane {statistics.median(latencies_ms):.0f} ms, "
          f"max {max(latencies_ms):.0f} ms")
    print(f"RAM système disponible après test: "
          f"{psutil.virtual_memory().available / 1_000_000_000:.2f} Go")


if __name__ == "__main__":
    main()