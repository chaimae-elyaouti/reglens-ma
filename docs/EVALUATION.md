# Plan d'évaluation

Principe : **aucun chiffre n'est écrit à l'avance**. Tout sort de `eval/run_eval.py` sur TON jeu de questions.

## 1. Construire le jeu de questions (`eval/golden_set.jsonl`)

Une ligne JSON par question. Exemple de format (la référence juridique doit être vérifiée dans le PDF utilisé) :

```json
{"id": "ex-001", "question": "Quelle est la durée normale du travail par semaine ?", "lang": "fr", "answerable": true, "category": "duree_travail", "gold": [{"doc_id": "code-travail-adala", "article": "184"}], "notes": "Exemple de format ; confirmer le gold dans la source utilisée."}
```

- 60 questions : ~40 avec réponse, ~10 SANS réponse dans le corpus (impôts, droit pénal, autre pays...), ~10 en
  situation réelle (« mon employeur... ») qui demandent plusieurs articles (`gold` avec plusieurs entrées).
- Écris-les en LISANT le Code : la bonne source (`gold`) doit venir du texte, pas du système. Ne regarde pas les
  résultats du retriever pour choisir `gold` (sinon tu mesures ton système contre lui-même).
- Mélange formulations juridiques et langage courant (« virer » / « licencier »), et ajoute quelques questions en
  anglais ou en arabe si tu veux tester le multilingue.
- Fais relire 10 questions par quelqu'un d'autre : c'est la meilleure défense contre ton propre biais.
- `eval/golden_set.example.jsonl` est un exemple de format, pas un benchmark validé. Vérifie les références dans
  la source utilisée et remplace/complète les exemples avant d'interpréter les scores.

## 2. Métriques

| Métrique | Calcul | Où |
|---|---|---|
| Hit@K | au moins un bon article dans le top K | retrieval |
| Recall@K | part des bons articles retrouvés dans le top K | retrieval |
| Precision@K | part du top K qui est correct (faible par construction avec 1–2 bons articles) | retrieval |
| MRR | moyenne de 1/rang du premier bon article | retrieval |
| Citation correctness | part des articles cités qui sont dans `gold` | bout en bout |
| Abstention | accuracy, précision, rappel du refus ; refus à tort / refus manqués | bout en bout |
| Faithfulness | annotation manuelle 0/1 sur ~30 réponses (colonne `faithful` du CSV) | bout en bout |
| Latence | p50 / p95 retrieval et total | les deux |
| Coût | tokens par requête (tarif = 0 en gratuit, mais note les tokens) | bout en bout |

Un LLM-juge (Ragas, DeepEval) consomme ton quota gratuit et doit être validé contre ton annotation manuelle :
optionnel.

## 3. Expériences

| # | Question | Comment |
|---|---|---|
| E1 | Dense, BM25 ou hybride ? | `--modes dense,bm25,hybrid` |
| E2 | Chunking par article ou taille fixe ? | `build_index --strategy fixed --size 1000 --out index/fixed1000`, puis `--index index/v1 index/fixed1000` |
| E3 | L'en-tête de contexte aide-t-il ? | `build_index --no-header --out index/noheader` |
| E4 | Taille max d'article avant découpe | `--max-chars 800/1500/2500` |
| E5 | Autre modèle d'embeddings | change `EMBEDDING_MODEL` (+ préfixes) et reconstruis l'index |
| E6 | Reranker : gain vs latence | `USE_RERANKER=true`, compare Hit@K et p95 |
| E7 | Seuil d'abstention | `--sweep` : choisis le compromis, note-le |
| E8 | Chroma ou pgvector vs numpy | optionnel : mêmes questions, compare résultats et latence |

## 4. Présenter les résultats

Dans le README : tableau daté (`eval/results/latest.md`) avec index, mode, nombre de questions, modèles ; 2–3 exemples
d'ÉCHECS analysés (pourquoi le bon article n'est pas remonté ?) ; les limites (taille du jeu, annotation par une seule
personne). Sur le CV : seulement des chiffres que tu as mesurés, avec la taille du jeu.
