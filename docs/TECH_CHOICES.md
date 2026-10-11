# Choix techniques et différences avec le tuto `rag-for-beginners`

| Tuto | Ici | Pourquoi |
|---|---|---|
| `OpenAIEmbeddings` (payant) | `sentence-transformers` en local (modèle multilingue) | Gratuit, tourne sur CPU, français/anglais |
| `ChatOpenAI` | SDK `openai` pointé sur Groq / Gemini (API compatibles) + bascule + cache | Gratuit, résilient aux quotas |
| LangChain partout | Code direct (numpy, rank_bm25, Pydantic) | Moins de dépendances, tests simples, déploiement léger, on voit ce qui se passe |
| Chroma | Matrice numpy + `chunks.jsonl` | Exact, sans serveur, compatible serverless (disque en lecture seule) |
| `CharacterTextSplitter` | Chunking par article + baseline fixe | L'article est l'unité de citation |
| `BM25Retriever` (espaces) | `rank_bm25` + tokenizer français | Accents, élisions, mots vides |
| RRF sur `page_content` | RRF sur identifiants | Deux textes identiques ne fusionnent pas à tort |
| Cohere rerank (anglais, clé) | Cross-encoder local multilingue, optionnel | Gratuit ; à mesurer |
| `print` | Logs JSON + `request_id` | Observabilité |
| Un script par technique | Modules + tests + CLI + API | Réutilisable, testable, déployable |
| Unstructured (hi_res) | PyMuPDF + nettoyage | Les lois sont du texte natif ; Unstructured = lourd (Poppler, Tesseract) |

## Parsing : quand passer à un parseur avancé ?

PyMuPDF suffit pour du texte natif. Si `quality.json` montre des articles manquants, des tableaux illisibles ou si le
PDF est un scan, teste Docling/Unstructured/OCR sur 3 documents et compare le rapport qualité et le Hit@K. Documente
le résultat même s'il est négatif.

## Points de maintenance

- Les noms de modèles, endpoints et quotas des fournisseurs évoluent. Vérifier leur documentation avant toute
  mise à jour de configuration ou de déploiement.
- Le rapport de construction d'index signale les chunks tronqués par le modèle d'embeddings. Contrôler ce rapport
  après chaque changement de modèle ou de corpus.
- Le reranker reste optionnel. Le garder désactivé tant qu'une évaluation comparative ne montre pas un gain utile.
