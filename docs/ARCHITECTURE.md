# Architecture

## Local

```
data/raw/*.pdf ──parse_pdf──► texte nettoyé ──chunking──► chunks.jsonl
                                                │ embeddings (sentence-transformers, CPU)
                                                ▼
                                   index/<version>/{chunks.jsonl, embeddings.npy, manifest.json, quality.json}
                                                │ chargé au démarrage
Question ─► FastAPI ─► Retriever (dense numpy + BM25 ─► RRF ─► filtres ─► [rerank])
                         │
                         ├─ évidence < seuil ? ─► abstention (LLM non appelé)
                         └─► prompt (citations par numéro d'article) ─► FailoverLLM (Groq ─► Gemini)
                                   ─► JSON ─► vérification des citations ─► AskResponse
```

## En ligne (piste AWS ; voir README du déploiement)

```
Navigateur ─► Frontend statique (S3 + CloudFront, ou GitHub Pages)
          ─► API (Lambda en conteneur + API Gateway / Function URL, ou Render)
                ├─ index chargé depuis l'image (lecture seule)
                ├─ embedding de la requête : modèle dans le conteneur
                ├─ LLM : API externe gratuite (clés = variables d'environnement)
                └─ logs JSON ─► CloudWatch
GitHub Actions : lint ─► tests ─► build image ─► déploiement
```

Explique chaque flèche en entretien : le front n'a aucun secret ; l'API est le seul composant qui connaît les
clés ; l'index est un artefact immuable produit hors ligne ; le LLM est interchangeable (interface `generate`).

## Modules

| Module | Rôle |
|---|---|
| `ingestion/parse_pdf.py` | Extraction PyMuPDF, suppression des en-têtes/pieds de page, nettoyage |
| `ingestion/chunking.py` | Détection des articles (numérotation monotone), sections, chunking article / taille fixe |
| `ingestion/quality.py` | Rapport qualité : articles manquants, chunks minuscules, doublons |
| `ingestion/build_index.py` | Embeddings avec cache, manifest versionné |
| `retrieval/*` | Tokenizer FR, BM25, dense, RRF, rerank optionnel |
| `generation/*` | Prompt, clients LLM (retry, bascule, cache), vérification des citations |
| `service.py` | Orchestration + garde-fous |
| `api/*` | FastAPI, CORS, limitation de débit, `request_id`, erreurs HTTP précises |
| `evaluation/metrics.py` | Hit@K, Recall@K, Precision@K, MRR, citation, abstention, seuil |
