# RegLens MA — recherche documentaire sur le droit du travail marocain

> Pose une question sur le Code du travail (loi n° 65-99). Le système retrouve les articles pertinents,
> répond **uniquement** à partir d'eux, **cite** chaque article, te laisse **relire le texte source**,
> et **refuse de répondre** quand les preuves sont insuffisantes.
>
> ⚠️ Outil de recherche documentaire, **pas un avis juridique**. La loi évolue (une réforme du Code du travail
> est annoncée : voir `docs/CORPUS.md`). Pour ta situation, consulte l'inspection du travail, un syndicat ou un avocat.

## Ce qui rend ce projet différent d'un « chat avec un PDF »

| Choix | Pourquoi |
|---|---|
| Un chunk **par article** (+ baseline à taille fixe pour comparer) | L'article est l'unité de citation en droit |
| Recherche **hybride** (embeddings + BM25 français + RRF) | Sens ET termes exacts ("préavis", "article 39") |
| **Abstention** calibrée sur ton jeu de questions | Un assistant juridique doit savoir dire « je ne sais pas » |
| Citations **vérifiées côté serveur** | Le LLM ne peut pas citer un passage qu'il n'a pas reçu |
| **Évaluation reproductible** (Hit@K, MRR, citation, abstention, latence) | Chaque choix technique est mesuré, pas supposé |
| Index **versionné** (manifest + hash du corpus) | Reproductibilité et traçabilité des résultats |
| Logs JSON, `request_id`, limitation de débit, pas de questions dans les logs | Robustesse et vie privée |

## Architecture (voir `docs/ARCHITECTURE.md`)

```
PDF officiel → parse/clean → chunks par article + métadonnées → embeddings locaux → index/ (versionné)
                                                                                         │
Question → API FastAPI → retrieval hybride (dense + BM25 → RRF) → garde-fou d'abstention → LLM (Groq/Gemini)
                                      → vérification des citations → réponse + passages numérotés → frontend
```

## Démarrage (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[ml,api,dev]"        # [ml] installe PyTorch : plusieurs centaines de Mo
copy .env.example .env                 # puis renseigne GROQ_API_KEY / GEMINI_API_KEY

pytest                                 # 41 tests, sans réseau, sans modèle
python scripts/fetch_corpus.py         # télécharge le PDF listé dans data/sources.json
python -m reglens.ingestion.build_index
type index\v1\quality.json             # LIS le rapport qualité avant de continuer

python -m reglens.cli ask "Quelle est la durée normale du travail ?" --no-llm   # retrieval seul, sans clé
python -m reglens.cli ask "Quelle est la durée normale du travail ?"            # avec le LLM

uvicorn reglens.api.app:create_app --factory --reload   # API sur http://localhost:8000/docs
python -m http.server 8080 -d frontend                  # frontend sur http://localhost:8080
```

## Évaluation (voir `docs/EVALUATION.md`)

```powershell
copy eval\golden_set.example.jsonl eval\golden_set.jsonl   # puis écris TES questions
python eval/run_eval.py --modes dense,bm25,hybrid
python eval/run_eval.py --sweep
```

## Résultats

_À remplir avec tes mesures, jamais avec des chiffres supposés._ Colle ici `eval/results/latest.md`
et ton analyse d'erreurs.

## Limites connues

_À remplir honnêtement (corpus limité au Code du travail, pas de jurisprudence, version du texte, langues...)._
