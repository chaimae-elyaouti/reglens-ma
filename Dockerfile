# Image de l'API. À construire UNE fois avant déploiement (en local, Docker Desktop consomme de la RAM :
# limite-le avec %UserProfile%\.wslconfig) ou, mieux, laisse GitHub Actions le faire.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf

# PyTorch version CPU : beaucoup plus petite que la version GPU par défaut.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[ml,api]"

# L'index est un artefact en lecture seule, construit hors ligne (voir README).
COPY index ./index
# Pré-télécharge le modèle d'embeddings dans l'image : pas de téléchargement au démarrage à froid.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

ENV INDEX_DIR=index/v1
EXPOSE 8000
CMD ["uvicorn", "reglens.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
