# Déploiement

Le projet n'a pas encore de déploiement public configuré. Les options ci-dessous sont des pistes ; avant d'en
choisir une, vérifier sa documentation, ses tarifs, ses quotas et ses exigences actuelles.

## Principes valables partout
- L'API est un conteneur (voir `Dockerfile`) ; l'index est copié dans l'image (lecture seule).
- Les clés (`GROQ_API_KEY`, `GEMINI_API_KEY`) sont des **variables d'environnement** de la plateforme, jamais dans
  l'image ni dans Git.
- `ALLOWED_ORIGINS` = l'URL exacte du frontend. `RATE_LIMIT_PER_IP_PER_DAY` protège ton quota LLM.
- `/health` pour la supervision, `/version` pour savoir quel index tourne.
- `LOG_QUESTIONS=false` en production (les questions peuvent contenir des informations personnelles).

## Option AWS
Lambda en conteneur (image dans un registre AWS : vérifie le coût de stockage) + API Gateway ou Function URL ;
frontend sur S3 + CloudFront ; logs dans CloudWatch (rétention courte) ; **AWS Budgets** avec alerte dès le premier jour.
Points d'attention : le démarrage à froid charge le modèle d'embeddings (mesure-le) ; la mémoire allouée détermine la
puissance CPU ; le disque est en lecture seule (c'est pourquoi l'index est un fichier numpy et non une base qui écrit).
À éviter pour rester gratuit : OpenSearch Serverless, NAT Gateway, Aurora.

## Option sans AWS
Backend sur un hébergeur de conteneurs à offre gratuite (par exemple Render : mise en veille après inactivité, d'où le
message « le serveur se réveille » du frontend), frontend statique (GitHub Pages ou équivalent).
Ajoute une vidéo démo au README : si le backend dort, la démo reste visible.

## Docker Desktop sur 8 Go
Crée `%UserProfile%\.wslconfig` :
```
[wsl2]
memory=3GB
processors=2
```
Puis `wsl --shutdown`. Ou construis l'image dans GitHub Actions.
