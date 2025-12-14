"""
Vector Store pour le RAG
Supporte ChromaDB et PostgreSQL avec pgvector
"""
import os
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

from .config import RAGConfig
from .chunker import Chunk
from .embeddings import EmbeddingProvider


@dataclass
class SearchResult:
    """Résultat d'une recherche dans le vector store"""
    content: str
    metadata: dict
    score: float
    chunk_id: str


class VectorStore:
    """
    Vector Store abstrait avec support ChromaDB et pgvector
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.embedding_provider = EmbeddingProvider(self.config)
        self._store = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """S'assure que le store est initialisé"""
        if self._initialized:
            return
        
        if self.config.use_pgvector:
            self._init_pgvector()
        else:
            self._init_chromadb()
        
        self._initialized = True
    
    def _init_chromadb(self):
        """Initialise ChromaDB"""
        import chromadb
        from chromadb.config import Settings
        
        # Créer le dossier si nécessaire
        os.makedirs(self.config.vector_store_path, exist_ok=True)
        
        self._client = chromadb.PersistentClient(
            path=self.config.vector_store_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Obtenir ou créer la collection
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _init_pgvector(self):
        """Initialise PostgreSQL avec pgvector"""
        import psycopg2
        from psycopg2.extras import execute_values
        
        # Connexion à PostgreSQL
        self._pg_conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "agent_simplon"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "password")
        )
        
        # Créer l'extension et la table si nécessaire
        with self._pg_conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            dimension = self.embedding_provider.get_embedding_model().dimension
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.collection_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({dimension})
                )
            """)
            
            # Créer un index HNSW pour la recherche rapide
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.config.collection_name}_embedding_idx
                ON {self.config.collection_name}
                USING hnsw (embedding vector_cosine_ops)
            """)
            
            self._pg_conn.commit()
    
    def add_chunks(self, chunks: List[Chunk]) -> List[str]:
        """
        Ajoute des chunks au vector store
        Retourne les IDs des chunks ajoutés
        """
        self._ensure_initialized()
        
        if not chunks:
            return []
        
        # Générer les embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_provider.embed_texts(texts)
        
        # Générer des IDs uniques
        import uuid
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        
        if self.config.use_pgvector:
            return self._add_pgvector(chunks, embeddings, chunk_ids)
        else:
            return self._add_chromadb(chunks, embeddings, chunk_ids)
    
    def _add_chromadb(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        chunk_ids: List[str]
    ) -> List[str]:
        """Ajoute des chunks à ChromaDB"""
        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks]
        )
        return chunk_ids
    
    def _add_pgvector(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        chunk_ids: List[str]
    ) -> List[str]:
        """Ajoute des chunks à PostgreSQL avec pgvector"""
        from psycopg2.extras import execute_values
        
        data = [
            (
                chunk_id,
                chunk.content,
                json.dumps(chunk.metadata),
                embedding
            )
            for chunk_id, chunk, embedding in zip(chunk_ids, chunks, embeddings)
        ]
        
        with self._pg_conn.cursor() as cursor:
            execute_values(
                cursor,
                f"""
                INSERT INTO {self.config.collection_name} (chunk_id, content, metadata, embedding)
                VALUES %s
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                data,
                template="(%s, %s, %s, %s::vector)"
            )
            self._pg_conn.commit()
        
        return chunk_ids
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Recherche les chunks les plus similaires à la requête
        """
        self._ensure_initialized()
        
        top_k = top_k or self.config.top_k
        
        # Générer l'embedding de la requête
        query_embedding = self.embedding_provider.embed_query(query)
        
        if self.config.use_pgvector:
            return self._search_pgvector(query_embedding, top_k, filter_metadata)
        else:
            return self._search_chromadb(query_embedding, top_k, filter_metadata)
    
    def _search_chromadb(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Recherche dans ChromaDB"""
        where = filter_metadata if filter_metadata else None
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=1 - results["distances"][0][i],  # Convertir distance en similarité
                    chunk_id=chunk_id
                ))
        
        return search_results
    
    def _search_pgvector(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Recherche dans PostgreSQL avec pgvector"""
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        # Construire la requête avec filtres optionnels
        where_clause = ""
        params = [embedding_str, top_k]
        
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(f"metadata->>{key!r} = %s")
                params.insert(-1, value)
            where_clause = "WHERE " + " AND ".join(conditions)
        
        with self._pg_conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT chunk_id, content, metadata, 1 - (embedding <=> %s::vector) as similarity
                FROM {self.config.collection_name}
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, [embedding_str] + params[:-1] + [embedding_str, top_k])
            
            rows = cursor.fetchall()
        
        return [
            SearchResult(
                content=row[1],
                metadata=row[2] if row[2] else {},
                score=float(row[3]),
                chunk_id=row[0]
            )
            for row in rows
        ]
    
    def delete(self, chunk_ids: List[str]) -> None:
        """Supprime des chunks par leurs IDs"""
        self._ensure_initialized()
        
        if self.config.use_pgvector:
            with self._pg_conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.config.collection_name} WHERE chunk_id = ANY(%s)",
                    (chunk_ids,)
                )
                self._pg_conn.commit()
        else:
            self._collection.delete(ids=chunk_ids)
    
    def clear(self) -> None:
        """Supprime tous les documents du vector store"""
        self._ensure_initialized()
        
        if self.config.use_pgvector:
            with self._pg_conn.cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {self.config.collection_name}")
                self._pg_conn.commit()
        else:
            # Recréer la collection
            self._client.delete_collection(self.config.collection_name)
            self._collection = self._client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def count(self) -> int:
        """Retourne le nombre de documents dans le store"""
        self._ensure_initialized()
        
        if self.config.use_pgvector:
            with self._pg_conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {self.config.collection_name}")
                return cursor.fetchone()[0]
        else:
            return self._collection.count()
    
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Retourne toutes les métadonnées des chunks"""
        self._ensure_initialized()
        
        if self.config.use_pgvector:
            with self._pg_conn.cursor() as cursor:
                cursor.execute(f"SELECT metadata FROM {self.config.collection_name}")
                return [row[0] for row in cursor.fetchall()]
        else:
            result = self._collection.get(include=['metadatas'])
            return result.get('metadatas', [])
