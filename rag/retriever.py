"""
Retriever pour le RAG
Récupère les documents les plus pertinents
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .config import RAGConfig
from .vector_store import VectorStore, SearchResult


@dataclass
class RetrievedDocument:
    """Document récupéré par le retriever"""
    content: str
    metadata: dict
    score: float
    chunk_id: str
    rank: int


class Retriever:
    """
    Retriever pour récupérer les documents pertinents
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        config: Optional[RAGConfig] = None
    ):
        self.config = config or RAGConfig()
        self.vector_store = vector_store or VectorStore(self.config)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievedDocument]:
        """
        Récupère les documents les plus pertinents pour une requête
        
        Args:
            query: La requête de recherche
            top_k: Nombre de documents à retourner
            filter_metadata: Filtres sur les métadonnées
            score_threshold: Score minimum pour inclure un document
        
        Returns:
            Liste des documents récupérés, triés par pertinence
        """
        top_k = top_k or self.config.top_k
        
        print(f"🔍 Retriever: recherche pour '{query[:100]}' (top_k={top_k})")
        
        # Recherche dans le vector store
        search_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        print(f"📊 Retriever: {len(search_results)} résultats trouvés")
        if search_results:
            print(f"   - Meilleur score: {search_results[0].score:.4f}")
            print(f"   - Source: {search_results[0].metadata.get('source', 'N/A')}")
            print(f"   - Contenu (50 chars): {search_results[0].content[:50]}...")
        
        # Filtrer par score si spécifié
        if score_threshold is not None:
            search_results = [
                r for r in search_results
                if r.score >= score_threshold
            ]
        
        # Convertir en RetrievedDocument avec rang
        documents = [
            RetrievedDocument(
                content=result.content,
                metadata=result.metadata,
                score=result.score,
                chunk_id=result.chunk_id,
                rank=i + 1
            )
            for i, result in enumerate(search_results)
        ]
        
        return documents
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        context_window: int = 1
    ) -> List[RetrievedDocument]:
        """
        Récupère les documents avec leur contexte environnant
        (chunks adjacents dans le même document source)
        
        Args:
            query: La requête de recherche
            top_k: Nombre de documents à retourner
            context_window: Nombre de chunks adjacents à inclure
        
        Returns:
            Documents avec contexte étendu
        """
        documents = self.retrieve(query, top_k)
        
        # TODO: Implémenter la récupération du contexte adjacent
        # Cela nécessite de stocker les informations de position des chunks
        
        return documents
    
    def hybrid_retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        keyword_weight: float = 0.3
    ) -> List[RetrievedDocument]:
        """
        Recherche hybride combinant recherche sémantique et par mots-clés
        
        Args:
            query: La requête de recherche
            top_k: Nombre de documents à retourner
            keyword_weight: Poids de la recherche par mots-clés (0-1)
        
        Returns:
            Documents récupérés avec scores combinés
        """
        top_k = top_k or self.config.top_k
        
        # Recherche sémantique
        semantic_results = self.vector_store.search(query=query, top_k=top_k * 2)
        
        # TODO: Implémenter la recherche par mots-clés (BM25)
        # Pour l'instant, on utilise uniquement la recherche sémantique
        
        # Convertir en RetrievedDocument
        documents = [
            RetrievedDocument(
                content=result.content,
                metadata=result.metadata,
                score=result.score,
                chunk_id=result.chunk_id,
                rank=i + 1
            )
            for i, result in enumerate(semantic_results[:top_k])
        ]
        
        return documents
