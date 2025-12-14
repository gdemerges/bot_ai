"""
Configuration pour le module RAG
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    """Configuration centralisée pour le pipeline RAG"""
    
    # Provider pour les embeddings: "ollama", "openai", "huggingface"
    embedding_provider: Literal["ollama", "openai", "huggingface"] = "ollama"
    
    # Provider pour le LLM: "ollama", "openai"
    llm_provider: Literal["ollama", "openai"] = "ollama"
    
    # Modèles Ollama
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_embedding_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    ollama_llm_model: str = field(default_factory=lambda: os.getenv("OLLAMA_LLM_MODEL", "llama3.2"))
    
    # Modèles OpenAI (fallback)
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    openai_llm_model: str = field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"))
    
    # Modèles HuggingFace
    huggingface_embedding_model: str = field(default_factory=lambda: os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    
    # Configuration du chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Configuration du retriever
    top_k: int = 10  # Nombre de documents à récupérer avant reranking
    
    # Configuration du reranker
    rerank_top_k: int = 5  # Nombre de documents après reranking
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    use_reranker: bool = True
    
    # Configuration du vector store
    vector_store_path: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_PATH", "./data/vector_store"))
    collection_name: str = "documents"
    
    # PostgreSQL pour pgvector (optionnel)
    use_pgvector: bool = field(default_factory=lambda: os.getenv("USE_PGVECTOR", "false").lower() == "true")
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Crée une configuration à partir des variables d'environnement"""
        return cls(
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "ollama"),
            llm_provider=os.getenv("RAG_LLM_PROVIDER", "ollama"),
        )
