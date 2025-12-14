"""
Reranker pour le RAG
Réordonne les documents récupérés pour améliorer la pertinence
"""
from typing import List, Optional
from dataclasses import dataclass

from .config import RAGConfig
from .retriever import RetrievedDocument


class Reranker:
    """
    Reranker pour réordonner les documents récupérés
    Utilise un modèle cross-encoder pour scorer la pertinence
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._model = None
    
    def _load_model(self):
        """Charge le modèle de reranking"""
        if self._model is not None:
            return
        
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.config.reranker_model)
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        Réordonne les documents par pertinence
        
        Args:
            query: La requête originale
            documents: Documents à réordonner
            top_k: Nombre de documents à retourner après reranking
        
        Returns:
            Documents réordonnés
        """
        if not documents:
            return []
        
        if not self.config.use_reranker:
            return documents[:top_k] if top_k else documents
        
        self._load_model()
        
        top_k = top_k or self.config.rerank_top_k
        
        # Préparer les paires query-document
        pairs = [(query, doc.content) for doc in documents]
        
        # Scorer les paires
        scores = self._model.predict(pairs)
        
        # Créer les nouveaux documents avec les scores mis à jour
        reranked_docs = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            reranked_docs.append(RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=float(score),
                chunk_id=doc.chunk_id,
                rank=doc.rank  # Garder le rang original pour référence
            ))
        
        # Trier par score décroissant
        reranked_docs.sort(key=lambda x: x.score, reverse=True)
        
        # Mettre à jour les rangs
        for i, doc in enumerate(reranked_docs):
            doc.rank = i + 1
        
        return reranked_docs[:top_k]


class LLMReranker:
    """
    Reranker utilisant un LLM pour scorer la pertinence
    Plus lent mais potentiellement plus précis
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        Réordonne les documents en utilisant un LLM
        """
        if not documents:
            return []
        
        top_k = top_k or self.config.rerank_top_k
        
        # Utiliser le LLM pour scorer chaque document
        scored_docs = []
        
        for doc in documents:
            score = self._score_document(query, doc.content)
            scored_docs.append(RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=score,
                chunk_id=doc.chunk_id,
                rank=doc.rank
            ))
        
        # Trier par score décroissant
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        
        # Mettre à jour les rangs
        for i, doc in enumerate(scored_docs):
            doc.rank = i + 1
        
        return scored_docs[:top_k]
    
    def _score_document(self, query: str, document: str) -> float:
        """
        Score un document par rapport à une requête en utilisant un LLM
        """
        import httpx
        
        prompt = f"""Sur une échelle de 0 à 10, évaluez la pertinence du document suivant par rapport à la question.
Répondez uniquement avec un nombre entre 0 et 10.

Question: {query}

Document: {document}

Score de pertinence:"""
        
        if self.config.llm_provider == "ollama":
            response = httpx.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.ollama_llm_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json().get("response", "5")
        else:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.config.openai_llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            result = response.choices[0].message.content
        
        # Extraire le score
        try:
            score = float(result.strip().split()[0])
            return min(max(score / 10.0, 0.0), 1.0)  # Normaliser entre 0 et 1
        except (ValueError, IndexError):
            return 0.5  # Score par défaut
