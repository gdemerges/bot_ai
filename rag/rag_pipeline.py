"""
Pipeline RAG complet
Orchestre tous les composants du RAG
"""
import os
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import httpx

from .config import RAGConfig
from .chunker import DocumentChunker, Chunk
from .embeddings import EmbeddingProvider
from .vector_store import VectorStore
from .retriever import Retriever, RetrievedDocument
from .reranker import Reranker


@dataclass
class RAGResponse:
    """Réponse du pipeline RAG"""
    answer: str
    sources: List[RetrievedDocument]
    query: str
    context_used: str


class RAGPipeline:
    """
    Pipeline RAG complet avec:
    - Chunking de documents
    - Embeddings (Ollama/OpenAI/HuggingFace)
    - Vector Store (ChromaDB/pgvector)
    - Retriever
    - Reranker
    - Génération de réponse (Ollama/OpenAI)
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig.from_env()
        
        # Initialiser les composants
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.embedding_provider = EmbeddingProvider(self.config)
        self.vector_store = VectorStore(self.config)
        self.retriever = Retriever(self.vector_store, self.config)
        self.reranker = Reranker(self.config)
    
    # ==================== INGESTION ====================
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> List[str]:
        """
        Ajoute un document au vector store
        
        Args:
            content: Contenu du document
            metadata: Métadonnées optionnelles
            source: Source du document (nom de fichier, URL, etc.)
        
        Returns:
            Liste des IDs des chunks créés
        """
        if metadata is None:
            metadata = {}
        
        if source:
            metadata["source"] = source
        
        # Chunker le document
        chunks = self.chunker.chunk_text(content, metadata)
        
        # Ajouter au vector store
        chunk_ids = self.vector_store.add_chunks(chunks)
        
        return chunk_ids
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Ajoute plusieurs documents au vector store
        
        Args:
            documents: Liste de dicts avec 'content', 'metadata' optionnel, 'source' optionnel
        
        Returns:
            Liste des IDs des chunks créés
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = doc.get("source")
            
            if source:
                metadata["source"] = source
            
            chunks = self.chunker.chunk_text(content, metadata)
            all_chunks.extend(chunks)
        
        return self.vector_store.add_chunks(all_chunks)
    
    def add_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Ajoute un fichier au vector store
        Supporte: .txt, .pdf, .docx, .md
        
        Args:
            file_path: Chemin vers le fichier
            metadata: Métadonnées optionnelles
        
        Returns:
            Liste des IDs des chunks créés
        """
        content = self._load_file(file_path)
        
        if metadata is None:
            metadata = {}
        
        metadata["source"] = os.path.basename(file_path)
        metadata["file_path"] = file_path
        
        return self.add_document(content, metadata, source=file_path)
    
    def _load_file(self, file_path: str) -> str:
        """Charge le contenu d'un fichier"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".txt" or ext == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                raise ImportError("pypdf est requis pour lire les fichiers PDF. Installez-le avec: pip install pypdf")
        
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                text = ""
                for para in doc.paragraphs:
                    text += para.text + "\n"
                return text
            except ImportError:
                raise ImportError("python-docx est requis pour lire les fichiers DOCX. Installez-le avec: pip install python-docx")
        
        else:
            # Essayer de lire comme texte
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    
    # ==================== RETRIEVAL ====================
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None
    ) -> List[RetrievedDocument]:
        """
        Récupère les documents pertinents pour une requête
        
        Args:
            query: La requête
            top_k: Nombre de documents à retourner
            use_reranker: Utiliser le reranker (par défaut: config)
        
        Returns:
            Liste des documents pertinents
        """
        # Récupérer plus de documents si on utilise le reranker
        retrieval_k = self.config.top_k if use_reranker is None else (
            self.config.top_k if use_reranker else top_k or self.config.rerank_top_k
        )
        
        # Retrieval initial
        documents = self.retriever.retrieve(query, top_k=retrieval_k)
        
        # Reranking si activé
        should_rerank = use_reranker if use_reranker is not None else self.config.use_reranker
        if should_rerank and documents:
            documents = self.reranker.rerank(
                query,
                documents,
                top_k=top_k or self.config.rerank_top_k
            )
        
        return documents
    
    # ==================== GENERATION ====================
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        system_prompt: Optional[str] = None
    ) -> RAGResponse:
        """
        Exécute une requête RAG complète
        
        Args:
            query: La question de l'utilisateur
            top_k: Nombre de documents à utiliser
            use_reranker: Utiliser le reranker
            system_prompt: Prompt système personnalisé
        
        Returns:
            RAGResponse avec la réponse et les sources
        """
        # Récupérer les documents pertinents
        documents = self.retrieve(query, top_k, use_reranker)
        
        if not documents:
            return RAGResponse(
                answer="Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question.",
                sources=[],
                query=query,
                context_used=""
            )
        
        # Construire le contexte
        context = self._build_context(documents)
        
        # Générer la réponse
        answer = self._generate_answer(query, context, system_prompt)
        
        return RAGResponse(
            answer=answer,
            sources=documents,
            query=query,
            context_used=context
        )
    
    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        """Construit le contexte à partir des documents récupérés"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", doc.metadata.get("original_filename", "Document inconnu"))
            context_parts.append(f"[Document {i}: {source}]\n{doc.content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Génère une réponse en utilisant le LLM configuré"""
        
        default_system_prompt = """Tu es un assistant strict qui répond UNIQUEMENT avec les informations du contexte.

RÈGLES ABSOLUES:
1. Lis ATTENTIVEMENT le contexte fourni
2. Si l'information est dans le contexte → Réponds avec cette information
3. TOUJOURS mentionner le nom du fichier source (ex: "Selon le document Seance_4_Culture_des_Entreprises.pdf...")
4. Si l'information n'est PAS dans le contexte → Réponds "Je n'ai pas trouvé cette information dans les documents."
5. Ne JAMAIS utiliser tes connaissances générales
6. Ne JAMAIS inventer d'informations
7. Réponds en français, de manière claire et concise"""
        
        system = system_prompt or default_system_prompt
        
        user_prompt = f"""CONTEXTE DISPONIBLE:
---
{context}
---

QUESTION: {query}

INSTRUCTIONS: Lis le contexte ci-dessus et réponds à la question en utilisant UNIQUEMENT les informations qu'il contient. Mentionne TOUJOURS le nom du fichier d'où provient l'information (ex: "Selon le document [nom_fichier]..." ou "D'après [nom_fichier]...").

RÉPONSE:"""
        
        if self.config.llm_provider == "ollama":
            return self._generate_ollama(system, user_prompt)
        else:
            return self._generate_openai(system, user_prompt)
    
    def _generate_ollama(self, system: str, user_prompt: str) -> str:
        """Génère une réponse avec Ollama"""
        response = httpx.post(
            f"{self.config.ollama_base_url}/api/chat",
            json={
                "model": self.config.ollama_llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False
            },
            timeout=120.0
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    
    def _generate_openai(self, system: str, user_prompt: str) -> str:
        """Génère une réponse avec OpenAI"""
        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model=self.config.openai_llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    # ==================== UTILITIES ====================
    
    def clear(self) -> None:
        """Vide le vector store"""
        self.vector_store.clear()
    
    def count(self) -> int:
        """Retourne le nombre de chunks dans le store"""
        return self.vector_store.count()
    
    def delete_by_source(self, source: str) -> None:
        """Supprime tous les chunks d'une source donnée"""
        # TODO: Implémenter la suppression par métadonnée
        pass
