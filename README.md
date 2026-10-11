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

pytest
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

### Mesures locales du 11 octobre 2026

Résultats mesurés sur l'index `index/v1` (version `7851e0c804`), avec `intfloat/multilingual-e5-small`
et le jeu local `eval/regression.jsonl` : **3 questions, toutes répondables**. Cet échantillon est trop
petit et trop homogène pour représenter la qualité générale du système. Le jeu et les rapports JSON
ne sont pas publiés dans ce dépôt.

| Mode | Hit@1 | Hit@3 | Hit@5 | Recall@10 | MRR | Latence retrieval p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0,667 | 1,000 | 1,000 | 0,574 | 0,833 | 45,2 / 60,1 |
| BM25 | 0,000 | 1,000 | 1,000 | 0,648 | 0,444 | 36,8 / 40,7 |
| Hybride | 0,667 | 0,667 | 1,000 | 0,694 | 0,750 | 42,8 / 46,5 |

Essai de bout en bout sur les mêmes 3 questions, avec Groq `openai/gpt-oss-20b` : citations correctes
pour les **2 réponses produites** (100 %), mais **1 abstention à tort sur 3 questions**. Les métriques
d'abstention ne sont pas représentatives, car le jeu ne contient aucune question qui devrait être refusée.
Latence totale p50 / p95 : **1 068,6 / 2 364,24 ms** ; moyenne : **1 935 tokens par question**.
La fidélité des réponses n'a pas encore été évaluée manuellement.

Tests : **43 passés** (`pytest -q`, 2,07 s). Mesure locale sans LLM ni reranker : démarrage **42,3 s**,
RAM au repos **1 266 Mo**, pic **1 311 Mo** ; sur 10 requêtes retrieval, médiane **32 ms**, maximum
**38 ms**. Ces temps dépendent de la machine et, pour le test bout en bout, du réseau et du fournisseur.

## Limites connues

- Le corpus évalué est limité à la version française du Code du travail (loi n° 65-99) ; il n'inclut pas
  la jurisprudence ni l'ensemble des textes réglementaires auxquels le Code renvoie.
- La date d'effet/version juridique du texte indexé n'est pas renseignée et son actualité n'a pas été
  vérifiée. Ne pas utiliser l'outil comme source juridique à jour ni comme avis juridique.
- Les résultats ci-dessus reposent sur 3 questions de régression locales seulement ; une question
  répondable a provoqué une abstention. Aucun score de fidélité manuelle ni performance générale n'est
  revendiqué.
