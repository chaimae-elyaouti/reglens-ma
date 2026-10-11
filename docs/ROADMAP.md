# Feuille de route jour par jour (≈ 17 jours à temps plein, +4 jours optionnels)

Règle : chaque jour se termine par un livrable commité sur Git (petit commit, message clair). Si un jour déborde,
ne saute pas le livrable : réduis le périmètre.

## Semaine 1 : fondations, corpus, retrieval mesuré

**Jour 1 : installation et prise en main.** `pip install -e ".[ml,api,dev]"`, `pytest` (tout doit passer), lis
`docs/ARCHITECTURE.md` et `docs/TECH_CHOICES.md`. Initialise Git, premier commit, dépôt GitHub (public ou privé, sans `.env`).
Mesure : RAM libre au repos (Gestionnaire des tâches). *Livrable : tests verts + CI verte.*

**Jour 2 : corpus et parsing.** `python scripts/fetch_corpus.py` (ou télécharge à la main si le lien a changé).
Ouvre le PDF, lis la table des matières et 10 articles. Lance `build_index` et lis `quality.json` : nombre d'articles
(~589 attendus), numéros manquants, chunks minuscules. Corrige `ARTICLE_RE` / nettoyage jusqu'à un rapport propre.
Compare les deux PDF de `sources.json` si besoin. *Livrable : rapport qualité propre, `as_of` renseigné.*

**Jour 3 : relecture des chunks + premier index.** Lis 30 chunks au hasard (`chunks.jsonl`) : articles coupés ?
sections correctes ? pages correctes ? Note les défauts dans `docs/INGESTION_NOTES.md`. Construis l'index complet et
note durée et pic de RAM (`build_index` ; Gestionnaire des tâches). Vérifie `chunks_truncated_by_model` dans `quality.json`.
*Livrable : index v1 + tableau « étape / durée / RAM ».*

**Jour 4 : explorer le retrieval.** `python -m reglens.cli ask "..." --no-llm` avec 20 questions en `dense`, `bm25`,
`hybrid`. Qu'est-ce que chaque mode rate ? *Concepts à relire : cosinus, BM25, RRF.* *Livrable : 1 page de constats.*

**Jour 5 : jeu d'évaluation (1/2).** Écris 30 questions en lisant le Code (pas en regardant les résultats). Format :
`docs/EVALUATION.md`. *Livrable : `eval/golden_set.jsonl` (30 questions).*

**Jour 6 : jeu d'évaluation (2/2) + premier tableau.** Complète à 60 (10 sans réponse, 10 situations réelles). Lance
`python eval/run_eval.py`. Analyse 10 échecs : mauvais chunking ? vocabulaire différent ? article renvoyant à un autre ?
*Livrable : premier `latest.md` + 10 échecs analysés.*

**Jour 7 : expériences retrieval.** E1 à E4 (`docs/EVALUATION.md`). Un index par variante. Note ce qui change. Un
résultat négatif est un résultat. *Livrable : tableau comparatif + 1 paragraphe de conclusion.*

## Semaine 2 : génération, API, qualité

**Jour 8 : clés LLM et génération.** Crée les clés Groq et Gemini, renseigne `.env`, **vérifie les noms de modèles**
dans leur doc. `python -m reglens.cli ask "..."` sur 10 questions. Lis le prompt (`generation/prompts.py`) et améliore-le
sur tes échecs (une modification à la fois). *Livrable : 10 réponses relues à la main.*

**Jour 9 : abstention.** `python eval/run_eval.py --sweep` : lis le tableau, choisis `MIN_EVIDENCE_SCORE`
(compromis refus à tort / refus manqués), note ton raisonnement. Teste les 10 questions sans réponse. *Livrable :
seuil choisi et justifié dans le README.*

**Jour 10 : évaluation de bout en bout.** `python eval/run_eval.py --with-llm --sleep 2.5` (réponses mises en cache).
Annote 30 réponses (colonne `faithful`), `--score-faithfulness`. Lis les cas où les citations sont fausses. *Livrable :
métriques de bout en bout + analyse d'erreurs.*

**Jour 11 : API robuste.** Lance `uvicorn ... --factory`, teste `/docs`, les erreurs (question vide/trop longue, filtre
invalide, clé LLM retirée → 503, limite de débit → 429). Lis les logs JSON : retrouve une requête par `request_id`.
Ajoute au moins 3 tests pour les bugs que tu as trouvés. *Livrable : tests supplémentaires + capture de logs.*

**Jour 12 : frontend.** Ouvre `frontend/index.html` via `python -m http.server`. Adapte les textes et les exemples
(situations réelles de salariés). Teste mobile (outils développeur du navigateur) et clavier. Ne dépasse pas la journée.
*Livrable : capture/GIF de la démo.*

## Semaine 3 : déploiement, finition, entretien

**Jour 13 : conteneur et CI.** Construis l'image une fois (limite WSL si Docker Desktop : `docs/DEPLOYMENT.md`) ou
laisse GitHub Actions le faire. CI verte (ruff + pytest). *Livrable : image qui répond sur `/health`.*

**Jour 14 : déploiement du backend.** Suis `docs/DEPLOYMENT.md` (piste AWS Lambda si ton compte est prêt, sinon
Render). Mets les clés en variables d'environnement, jamais dans l'image. Mesure le démarrage à froid.
*Livrable : URL d'API publique.*

**Jour 15 : déploiement du frontend + sécurité.** Héberge le front (S3+CloudFront ou GitHub Pages), règle `API_BASE`
et `ALLOWED_ORIGINS`, teste la limite de débit, alerte de budget. *Livrable : démo publique accessible sans compte.*

**Jour 16 : résultats finaux + documentation.** Réexécute toute l'évaluation sur l'index déployé, mets les tableaux
dans le README, rédige « Limites » honnêtement, dessine le diagramme d'architecture (Excalidraw/Mermaid). Enregistre une
vidéo démo de 2 minutes. *Livrable : README final.*

**Jour 17 : entretien et diffusion.** Prépare les réponses aux 15 questions (liste en bas du README plus tard), refais
l'architecture au tableau sans notes, publie un post LinkedIn, tag `v1.0`. *Livrable : tag + post.*

## Jours optionnels (après la V1)

- **J18** : comparer numpy et pgvector (Postgres, tu connais) : mêmes questions, résultats et latence.
- **J19** : reformulation de question avec historique (comme le fichier 4 du tuto), évaluée comme les autres.
- **J20** : ajouter un second texte (après vérification de sa version et de sa licence) avec ses questions d'évaluation.
- **J21** : gestion des versions de la loi (réforme) : nouveau `doc_id`, filtre par version, test de non-régression.

## Surveiller ton PC pendant le projet

| Moment | Risque | Réflexe |
|---|---|---|
| `build_index` (embeddings) | RAM et CPU au maximum | Ferme navigateur et VS Code inutiles, branche le secteur, essaie `EMBEDDING_BATCH_SIZE` 8/16/32 et note la durée |
| Modèle d'embeddings plus gros | Mémoire, lenteur | Ne change que si l'évaluation montre un gain |
| Reranker | Latence CPU par requête | Mesure p95 avant/après |
| Docker Desktop | WSL2 prend beaucoup de RAM | Limite-le (`.wslconfig`) ou construis dans GitHub Actions |
| Cache Hugging Face | Plusieurs Go sur le SSD | Supprime les modèles que tu n'utilises plus (`~/.cache/huggingface`) |
