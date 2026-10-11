"""Configuration centrale. Toutes les valeurs viennent de variables d'environnement / .env.

Pourquoi pydantic-settings ? Chaque réglage est typé et validé au démarrage (un entier mal écrit
fait échouer le programme tout de suite, pas au milieu d'une requête), et le MÊME code tourne en local
et en production : seules les variables d'environnement changent.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    log_level: str = "INFO"
    log_questions: bool = False

    # --- index ---
    index_dir: str = "index/v1"

    # --- embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16

    # --- retrieval ---
    top_k: int = 5
    candidates: int = 20
    rrf_k: int = 60
    min_evidence_score: float = 0.0
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # --- LLM (endpoints compatibles OpenAI) ---
    llm_providers: str = "groq,gemini"
    llm_timeout_s: float = 30.0
    llm_max_tokens: int = 700
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-2.5-flash"

    # --- API ---
    allowed_origins: str = "http://localhost:5173,http://localhost:8080"
    rate_limit_per_ip_per_day: int = 40  # 0 = désactivé
    max_question_chars: int = 600

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def provider_names(self) -> list[str]:
        return [p.strip().lower() for p in self.llm_providers.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
