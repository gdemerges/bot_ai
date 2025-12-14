FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier d'abord les requirements pour profiter du cache Docker
COPY requirements.txt .

# Installer les dépendances en plusieurs étapes pour un meilleur cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi uvicorn pydantic python-dotenv python-multipart requests httpx && \
    pip install --no-cache-dir \
    openai psycopg2-binary discord.py streamlit asyncpraw && \
    pip install --no-cache-dir \
    pandas langchain pypdf pytest prometheus-fastapi-instrumentator PyJWT python-docx numpy && \
    pip install --no-cache-dir \
    chromadb sentence-transformers

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
