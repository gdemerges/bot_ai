"""
Module d'embeddings pour le RAG
Supporte Ollama, OpenAI et HuggingFace
"""
import os
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np

from .config import RAGConfig


class BaseEmbedding(ABC):
    """Classe de base pour les embeddings"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour une liste de textes"""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Génère un embedding pour une requête"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Retourne la dimension des embeddings"""
        pass


class OllamaEmbedding(BaseEmbedding):
    """Embeddings via Ollama (local)"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension = None
        
        # Import httpx pour les appels async
        import httpx
        self.client = httpx.Client(timeout=60.0)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour plusieurs textes"""
        embeddings = []
        for text in texts:
            embedding = self._embed_single(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Génère un embedding pour une requête"""
        return self._embed_single(query)
    
    def _embed_single(self, text: str) -> List[float]:
        """Génère un embedding pour un seul texte"""
        response = self.client.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": text
            }
        )
        response.raise_for_status()
        data = response.json()
        # Ollama retourne 'embeddings' (pluriel) qui est une liste
        embeddings_list = data.get("embeddings", data.get("embedding", []))
        embedding = embeddings_list[0] if isinstance(embeddings_list, list) and len(embeddings_list) > 0 else embeddings_list
        
        if self._dimension is None and embedding:
            self._dimension = len(embedding)
        
        return embedding
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Générer un embedding test pour obtenir la dimension
            test_embedding = self._embed_single("test")
            self._dimension = len(test_embedding)
        return self._dimension


class OpenAIEmbedding(BaseEmbedding):
    """Embeddings via OpenAI API"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        
        # Dimensions connues pour les modèles OpenAI
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour plusieurs textes"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, query: str) -> List[float]:
        """Génère un embedding pour une requête"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        return response.data[0].embedding
    
    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.model, 1536)


class HuggingFaceEmbedding(BaseEmbedding):
    """Embeddings via HuggingFace (local)"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour plusieurs textes"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Génère un embedding pour une requête"""
        embedding = self.model.encode([query])[0]
        return embedding.tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingProvider:
    """
    Factory pour créer le bon provider d'embeddings
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._embedding_model = None
    
    def get_embedding_model(self) -> BaseEmbedding:
        """Retourne le modèle d'embedding configuré"""
        if self._embedding_model is not None:
            return self._embedding_model
        
        provider = self.config.embedding_provider
        
        if provider == "ollama":
            self._embedding_model = OllamaEmbedding(
                base_url=self.config.ollama_base_url,
                model=self.config.ollama_embedding_model
            )
        elif provider == "openai":
            self._embedding_model = OpenAIEmbedding(
                model=self.config.openai_embedding_model
            )
        elif provider == "huggingface":
            self._embedding_model = HuggingFaceEmbedding(
                model_name=self.config.huggingface_embedding_model
            )
        else:
            raise ValueError(f"Provider d'embedding inconnu: {provider}")
        
        return self._embedding_model
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Raccourci pour générer des embeddings"""
        return self.get_embedding_model().embed_texts(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """Raccourci pour générer un embedding de requête"""
        return self.get_embedding_model().embed_query(query)
