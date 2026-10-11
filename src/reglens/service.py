"""Orchestration d'une question : retrieval -> garde-fous -> LLM -> vérification -> réponse.

C'est ici que vit la logique "robuste" : on refuse de répondre quand les preuves sont insuffisantes
(sans même appeler le LLM, ce qui économise ton quota gratuit), et on rejette les réponses dont les
citations ne correspondent pas aux passages fournis.
"""

from __future__ import annotations

import hashlib
import logging
import re
from time import perf_counter

from reglens.config import Settings
from reglens.generation.citations import parse_llm_json, validate_article_citations
from reglens.generation.prompts import build_messages
from reglens.logging_setup import log_event, request_id_var
from reglens.retrieval.retriever import Retriever
from reglens.schemas import AskRequest, AskResponse, Hit, Passage

ARTICLE_REFERENCE_RE = re.compile(
    r"\barticles?\s+(\d+(?:\s*(?:,|;|et|ou|à|au)\s*\d+)*)", re.IGNORECASE
)
MAX_REFERENCED_ARTICLES = 5
EXCEPTION_REFERENCE_RE = re.compile(
    r"\b(?:sous réserve|dérogations?)\b[^.!?]{0,180}\barticles?\s+"
    r"(\d+(?:\s*(?:,|;|et|ou|à|au)\s*\d+)*)",
    re.IGNORECASE,
)


def make_disclaimer(as_of: str) -> str:
    version = f"version du {as_of}" if as_of else "date de version non renseignée"
    return (
        f"Information juridique générale fondée sur les textes indexés ({version}). La loi peut avoir "
        "changé depuis. Ce n'est pas un avis juridique : pour votre situation, consultez l'inspection du "
        "travail, un syndicat ou un avocat."
    )


def to_passages(hits: list[Hit]) -> list[Passage]:
    return [
        Passage(
            n=i,
            chunk_id=h.chunk.id,
            doc_id=h.chunk.doc_id,
            doc_title=h.chunk.doc_title,
            article=h.chunk.article,
            section_path=h.chunk.section_path,
            page=h.chunk.page,
            score=round(h.score, 5),
            dense_score=None if h.dense_score is None else round(h.dense_score, 4),
            text=h.chunk.text,
        )
        for i, h in enumerate(hits, 1)
    ]


def expand_article_references(
    retriever: Retriever, hits: list[Hit], filters: dict[str, str] | None
) -> list[Hit]:
    if not hits or (filters and "article" in filters):
        return hits

    expanded = list(hits)
    seen_chunks = {hit.chunk.id for hit in hits}
    seen_references: set[tuple[str, str]] = set()
    chunks_by_article: dict[tuple[str, str], list] = {}
    for chunk in retriever.index.chunks:
        for article in chunk.articles():
            chunks_by_article.setdefault((chunk.doc_id, article), []).append(chunk)

    for hit in hits:
        for match in ARTICLE_REFERENCE_RE.finditer(hit.chunk.text):
            for article in re.findall(r"\d+", match.group(1)):
                reference = (hit.chunk.doc_id, article)
                if article == hit.chunk.article or reference in seen_references:
                    continue
                seen_references.add(reference)
                if len(seen_references) > MAX_REFERENCED_ARTICLES:
                    return expanded

                for chunk in chunks_by_article.get(reference, []):
                    if chunk.id in seen_chunks:
                        continue
                    if filters and filters.get("doc_id") and chunk.doc_id != filters["doc_id"]:
                        continue
                    if filters and filters.get("lang") and chunk.lang != filters["lang"]:
                        continue
                    expanded.append(Hit(chunk=chunk, score=0.0))
                    seen_chunks.add(chunk.id)

    return expanded


def add_exception_caveat(answer: str, source_hits: list[Hit], hits: list[Hit]) -> str:
    clauses: list[tuple[str, str, str, list[str]]] = []
    for hit in source_hits:
        normalized = re.sub(r"\s+", " ", hit.chunk.text).strip()
        for match in EXCEPTION_REFERENCE_RE.finditer(normalized):
            start = max(
                normalized.rfind(mark, 0, match.start()) for mark in ".!?"
            ) + 1
            end = normalized.find(".", match.end())
            clause = normalized[start : end + 1 if end >= 0 else len(normalized)].strip()
            references = list(dict.fromkeys(re.findall(r"\d+", match.group(1))))
            item = (hit.chunk.doc_id, hit.chunk.article, clause, references)
            if item not in clauses:
                clauses.append(item)

    if not clauses:
        return answer

    passages_by_article: dict[tuple[str, str], list[Hit]] = {}
    for hit in hits:
        for article in hit.chunk.articles():
            passages_by_article.setdefault((hit.chunk.doc_id, article), []).append(hit)

    answer += "\n\nRéserve légale mentionnée dans le texte :"
    for doc_id, source_article, clause, references in clauses:
        answer += f"\n« {clause} » [Art. {source_article}]"
        missing = []
        for article in references:
            passages = passages_by_article.get((doc_id, article), [])
            if not passages:
                missing.append(article)
                continue
            for hit in passages:
                text = re.sub(r"\s+", " ", hit.chunk.text).strip()
                text = re.sub(rf"^Article\s+{re.escape(article)}\s*", "", text, flags=re.IGNORECASE)
                answer += f"\nArticle [Art. {article}] : « {text} »"
        if missing:
            answer += (
                f"\nLe texte des articles {', '.join(missing)} n'a pas été retrouvé dans le corpus ; "
                "leurs conditions ne peuvent donc pas être précisées ici."
            )
    return answer


class RagService:
    def __init__(self, retriever: Retriever, llm, settings: Settings):
        self.retriever, self.llm, self.settings = retriever, llm, settings
        self.manifest = retriever.index.manifest

    def get_chunk(self, chunk_id: str) -> dict | None:
        i = self.retriever.index.by_id.get(chunk_id)
        if i is None:
            return None
        chunks = self.retriever.index.chunks
        return {
            "chunk": chunks[i].model_dump(),
            "previous_id": chunks[i - 1].id if i > 0 else None,
            "next_id": chunks[i + 1].id if i + 1 < len(chunks) else None,
        }

    def ask(self, req: AskRequest, request_id: str = "") -> AskResponse:
        t0 = perf_counter()
        as_of = self.manifest.get("corpus_as_of", "")
        base = {
            "index_version": self.manifest.get("index_version", ""),
            "corpus_as_of": as_of,
            "disclaimer": make_disclaimer(as_of),
            "request_id": request_id,
        }
        result = self.retriever.retrieve(
            req.question, k=req.k, mode=req.mode, filters=req.filters, rerank=req.rerank
        )
        hits = expand_article_references(self.retriever, result.hits, req.filters)
        sources = to_passages(hits)
        timings = dict(result.timings_ms)

        def finish(resp: AskResponse) -> AskResponse:
            resp.timings_ms["total"] = round((perf_counter() - t0) * 1000, 1)
            resp.timings_ms = {k: round(v, 1) for k, v in resp.timings_ms.items()}
            self._log(req, resp)
            return resp

        def abstain(reason: str, **extra) -> AskResponse:
            return finish(
                AskResponse(
                    answer=None, abstain=True, abstain_reason=reason, sources=sources,
                    timings_ms=timings, evidence_score=result.evidence_score, **base, **extra,
                )
            )

        if not result.hits:
            return abstain("no_hits")
        thr = self.settings.min_evidence_score
        if thr > 0 and result.evidence_score is not None and result.evidence_score < thr:
            return abstain("low_evidence")  # le LLM n'est PAS appelé

        t_llm = perf_counter()
        llm_res = self.llm.generate(build_messages(req.question, hits))  # peut lever LLMUnavailable
        timings["llm"] = (perf_counter() - t_llm) * 1000
        llm_info = {
            "llm": f"{llm_res.provider}/{llm_res.model}",
            "tokens": {"prompt": llm_res.prompt_tokens, "completion": llm_res.completion_tokens},
        }

        parsed = parse_llm_json(llm_res.text)
        if parsed is None:
            return abstain("unparseable_output", **llm_info)
        answer = parsed.get("answer")
        if not answer:
            return abstain("llm_abstained", **llm_info)

        answer = add_exception_caveat(str(answer), result.hits, hits)
        allowed_articles = {article for hit in hits for article in hit.chunk.articles()}
        answer, cited, issues = validate_article_citations(
            answer, parsed.get("citations") or [], allowed_articles
        )
        if not cited:  # réponse non ancrée dans les sources : on ne la montre pas
            return abstain("ungrounded", issues=issues, **llm_info)
        return finish(
            AskResponse(
                answer=answer, abstain=False, cited=cited, sources=sources, issues=issues,
                timings_ms=timings, evidence_score=result.evidence_score, **llm_info, **base,
            )
        )

    def _log(self, req: AskRequest, resp: AskResponse) -> None:
        fields = {
            "question_len": len(req.question),
            "question_hash": hashlib.sha256(req.question.encode()).hexdigest()[:12],
            "mode": req.mode,
            "k": req.k,
            "abstain": resp.abstain,
            "abstain_reason": resp.abstain_reason,
            "evidence": None if resp.evidence_score is None else round(resp.evidence_score, 4),
            "cited": resp.cited,
            "issues": resp.issues,
            "timings_ms": resp.timings_ms,
            "tokens": resp.tokens,
            "llm": resp.llm,
            "index_version": resp.index_version,
        }
        if self.settings.log_questions:  # désactivé par défaut : vie privée
            fields["question"] = req.question
        log_event("ask", level=logging.INFO, **fields)


def build_service(settings: Settings, *, cache_llm: bool = False, with_llm: bool = True) -> RagService:
    """Assemble les composants réels (charge le modèle d'embeddings : plusieurs secondes)."""
    from reglens.generation.llm import build_llm
    from reglens.retrieval.embedder import SentenceTransformerEmbedder
    from reglens.retrieval.store import VectorIndex

    index = VectorIndex.load(settings.index_dir)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.query_prefix,
        settings.passage_prefix,
        settings.embedding_device,
        settings.embedding_batch_size,
    )
    reranker = None
    if settings.use_reranker:
        from reglens.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.reranker_model, settings.embedding_device)
    retriever = Retriever(
        index, embedder, candidates=settings.candidates, rrf_k=settings.rrf_k, reranker=reranker
    )
    llm = build_llm(settings, cache=cache_llm) if with_llm else None
    return RagService(retriever, llm, settings)


__all__ = ["RagService", "build_service", "make_disclaimer", "to_passages", "request_id_var"]
