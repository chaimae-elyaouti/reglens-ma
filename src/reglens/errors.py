"""Exceptions métier. L'API les traduit en codes HTTP précis (422, 503...)."""


class InvalidRequest(ValueError):
    """La requête est mal formée (filtre inconnu, question trop longue...). -> HTTP 422."""


class LLMUnavailable(RuntimeError):
    """Aucun fournisseur LLM n'a pu répondre (quota, panne, clé absente). -> HTTP 503."""
