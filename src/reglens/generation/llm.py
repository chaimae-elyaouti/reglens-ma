"""Clients LLM.

Idée clé : Groq ET Gemini exposent une API compatible OpenAI. Un seul client (le SDK `openai`) suffit :
on change juste base_url, clé et nom du modèle. Le code applicatif ne connaît que `generate(messages)`.

- OpenAICompatLLM : un fournisseur, avec retries (tenacity) sur erreurs transitoires (429, timeout, 5xx).
- FailoverLLM    : essaie les fournisseurs dans l'ordre. Les quotas gratuits changent sans prévenir :
                   ne dépends jamais d'un seul.
- CachedLLM      : cache disque (clé = hash du prompt). Pendant l'évaluation, tu ne consommes pas deux fois
                   ton quota et tes résultats sont reproductibles.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from reglens.config import Settings
from reglens.errors import LLMUnavailable
from reglens.logging_setup import log_event

RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


class OpenAICompatLLM:
    def __init__(self, name: str, base_url: str, api_key: str, model: str, timeout: float, max_tokens: int):
        self.name, self.model, self.max_tokens = name, model, max_tokens
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(RETRYABLE),
        reraise=True,
    )
    def generate(self, messages: list[dict]) -> LLMResult:
        t0 = perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=self.max_tokens
        )
        usage = getattr(resp, "usage", None)
        return LLMResult(
            text=resp.choices[0].message.content or "",
            provider=self.name,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=(perf_counter() - t0) * 1000,
        )


class FailoverLLM:
    def __init__(self, providers: list):
        self.providers = providers

    def generate(self, messages: list[dict]) -> LLMResult:
        errors = []
        for p in self.providers:
            try:
                return p.generate(messages)
            except openai.OpenAIError as e:
                errors.append(f"{p.name}:{type(e).__name__}")
                log_event("llm_provider_failed", level=logging.WARNING, provider=p.name, error=str(e)[:200])
        raise LLMUnavailable(f"Tous les fournisseurs LLM ont échoué ({', '.join(errors)}).")


class CachedLLM:
    def __init__(self, inner, cache_dir: str | Path):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def generate(self, messages: list[dict]) -> LLMResult:
        key = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.dir / f"{key}.json"
        if path.exists():
            return LLMResult(**json.loads(path.read_text(encoding="utf-8")))
        result = self.inner.generate(messages)
        path.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
        return result


def build_llm(settings: Settings, cache: bool = False):
    providers = []
    for name in settings.provider_names:
        key = getattr(settings, f"{name}_api_key", "")
        if not key:
            continue
        providers.append(
            OpenAICompatLLM(
                name,
                getattr(settings, f"{name}_base_url"),
                key,
                getattr(settings, f"{name}_model"),
                settings.llm_timeout_s,
                settings.llm_max_tokens,
            )
        )
    if not providers:
        raise LLMUnavailable("Aucune clé API configurée : renseigne GROQ_API_KEY ou GEMINI_API_KEY dans .env")
    llm = FailoverLLM(providers)
    return CachedLLM(llm, ".cache/llm") if cache else llm
