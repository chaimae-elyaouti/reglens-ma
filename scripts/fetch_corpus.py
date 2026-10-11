"""Télécharge les PDF listés (enabled=true) dans data/raw/ et enregistre leur empreinte SHA-256.

    python scripts/fetch_corpus.py

Les PDF ne sont pas versionnés dans Git : seul data/sources.json l'est, ce qui rend le corpus
reproductible (même URL, même empreinte) sans republier les textes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    docs = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))["documents"]
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    for d in docs:
        if not d.get("enabled"):
            continue
        dest = raw / d["local_file"]
        print(f"Téléchargement : {d['url']}")
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            r = client.get(d["url"])
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise SystemExit(f"{d['doc_id']} : le contenu n'est pas un PDF (type {r.headers.get('content-type')}).")
        dest.write_bytes(r.content)
        sha = hashlib.sha256(r.content).hexdigest()
        (raw / f"{d['local_file']}.meta.json").write_text(
            json.dumps({"url": d["url"], "sha256": sha, "retrieved_on": date.today().isoformat(),
                        "bytes": len(r.content)}, indent=2),
            encoding="utf-8",
        )
        print(f"  OK {dest.name} ({len(r.content) / 1e6:.1f} Mo) sha256={sha[:12]}…")


if __name__ == "__main__":
    main()
