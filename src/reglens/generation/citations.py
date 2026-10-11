"""Parsing de la sortie du LLM et validation des citations par article."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_ARTICLE_INLINE = re.compile(r"\[Art\.?\s*(\d{1,3})\]", re.IGNORECASE)


def parse_llm_json(text: str) -> dict | None:
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate_article_citations(
    answer: str | None, declared: list, allowed: set[str]
) -> tuple[str | None, list[str], list[str]]:
    """Ne conserve que les citations d’articles présents dans les passages fournis."""
    issues: list[str] = []
    allowed_articles = {str(article).strip() for article in allowed}
    declared_articles = {
        str(citation).strip()
        for citation in declared
        if isinstance(citation, (int, str)) and str(citation).strip().isdigit()
    }
    inline_articles = set(_ARTICLE_INLINE.findall(answer or ""))

    claimed = declared_articles | inline_articles
    invalid = claimed - allowed_articles
    if invalid:
        issues.append(f"citations_hors_passages:{sorted(invalid, key=int)}")
        if answer:
            answer = _ARTICLE_INLINE.sub(
                lambda match: "" if match.group(1) not in allowed_articles else match.group(0),
                answer,
            )
    cited = sorted(claimed & allowed_articles, key=int)
    if answer and not cited:
        issues.append("reponse_sans_citation")
    return answer, cited, issues
