"""Limiteur de requêtes par IP et par jour, en mémoire.

But : protéger la quota LLM gratuit contre un script ou un curieux. Limite connue : l'état est local à
UN processus. Sur Lambda (plusieurs instances) ou après un redémarrage, il est approximatif : pour un vrai
plafond, on utiliserait API Gateway (usage plans) ou une table partagée (DynamoDB). À mentionner en entretien.
"""

from __future__ import annotations

from datetime import date
from threading import Lock


class DailyIPLimiter:
    def __init__(self, limit: int):
        self.limit = limit
        self._counts: dict[tuple[str, date], int] = {}
        self._lock = Lock()

    def allow(self, ip: str) -> bool:
        if self.limit <= 0:
            return True
        today = date.today()
        with self._lock:
            for key in [k for k in self._counts if k[1] != today]:  # purge des jours passés
                del self._counts[key]
            n = self._counts.get((ip, today), 0)
            if n >= self.limit:
                return False
            self._counts[(ip, today)] = n + 1
            return True
