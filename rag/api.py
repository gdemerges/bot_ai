"""
Endpoints API pour le module RAG
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile

from rag import RAGPipeline, RAGConfig

router = APIRouter(prefix="/rag", tags=["RAG"])

# Instance globale du pipeline RAG
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Retourne l'instance du pipeline RAG (lazy loading)"""
    global _rag_pipeline
    if _rag_pipeline is None:
        config = RAGConfig.from_env()
        _rag_pipeline = RAGPipeline(config)
    return _rag_pipeline


# ==================== MODELS ====================

class RAGQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    use_reranker: Optional[bool] = None
    system_prompt: Optional[str] = None


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query: str


class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


class DocumentsRequest(BaseModel):
    documents: List[DocumentRequest]


class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    use_reranker: Optional[bool] = None


class RetrieveResponse(BaseModel):
    documents: List[Dict[str, Any]]
    query: str


class StatsResponse(BaseModel):
    total_chunks: int
    embedding_provider: str
    llm_provider: str
    vector_store_type: str


# ==================== ENDPOINTS ====================

@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Exécute une requête RAG complète
    
    - Récupère les documents pertinents
    - Applique le reranking si activé
    - Génère une réponse avec le LLM
    """
    try:
        pipeline = get_rag_pipeline()
        response = pipeline.query(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            system_prompt=request.system_prompt
        )
        
        return RAGQueryResponse(
            answer=response.answer,
            sources=[
                {
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score,
                    "rank": doc.rank
                }
                for doc in response.sources
            ],
            query=response.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la requête RAG: {str(e)}")


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """
    Récupère les documents pertinents sans générer de réponse
    Utile pour le debug ou l'inspection des résultats
    """
    try:
        pipeline = get_rag_pipeline()
        documents = pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker
        )
        
        return RetrieveResponse(
            documents=[
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score,
                    "rank": doc.rank,
                    "chunk_id": doc.chunk_id
                }
                for doc in documents
            ],
            query=request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération: {str(e)}")


@router.post("/documents")
async def add_document(request: DocumentRequest):
    """
    Ajoute un document texte au vector store
    """
    try:
        pipeline = get_rag_pipeline()
        chunk_ids = pipeline.add_document(
            content=request.content,
            metadata=request.metadata,
            source=request.source
        )
        
        return {
            "message": "Document ajouté avec succès",
            "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'ajout du document: {str(e)}")


@router.post("/documents/batch")
async def add_documents_batch(request: DocumentsRequest):
    """
    Ajoute plusieurs documents au vector store
    """
    try:
        pipeline = get_rag_pipeline()
        documents = [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "source": doc.source
            }
            for doc in request.documents
        ]
        
        chunk_ids = pipeline.add_documents(documents)
        
        return {
            "message": f"{len(request.documents)} documents ajoutés avec succès",
            "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'ajout des documents: {str(e)}")


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Upload un fichier et l'ajoute au vector store
    Supporte: .txt, .pdf, .docx, .md
    """
    try:
        # Sauvegarder le fichier temporairement
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            pipeline = get_rag_pipeline()
            
            # Parser les métadonnées si fournies
            meta = {}
            if metadata:
                import json
                meta = json.loads(metadata)
            
            # Ajouter le nom original du fichier dans les métadonnées
            meta["original_filename"] = file.filename
            meta["source"] = file.filename
            
            chunk_ids = pipeline.add_file(tmp_path, meta)
            
            return {
                "message": f"Fichier '{file.filename}' ajouté avec succès",
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids
            }
        finally:
            # Nettoyer le fichier temporaire
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'upload: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Retourne les statistiques du vector store
    """
    try:
        pipeline = get_rag_pipeline()
        config = pipeline.config
        
        return StatsResponse(
            total_chunks=pipeline.count(),
            embedding_provider=config.embedding_provider,
            llm_provider=config.llm_provider,
            vector_store_type="pgvector" if config.use_pgvector else "chromadb"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des stats: {str(e)}")


@router.get("/files")
async def list_files():
    """
    Liste tous les fichiers indexés dans le vector store
    Retourne les noms de fichiers uniques avec le nombre de chunks par fichier
    """
    try:
        pipeline = get_rag_pipeline()
        
        # Récupérer toutes les métadonnées
        all_metadata = pipeline.vector_store.get_all_metadata()
        
        # Grouper par source/fichier
        files_info = {}
        for metadata in all_metadata:
            source = metadata.get('source') or metadata.get('original_filename', 'Unknown')
            if source not in files_info:
                files_info[source] = {
                    'filename': source,
                    'chunk_count': 0
                }
            files_info[source]['chunk_count'] += 1
        
        return {
            "total_files": len(files_info),
            "files": list(files_info.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des fichiers: {str(e)}")


@router.delete("/clear")
async def clear_vector_store():
    """
    Vide le vector store (ATTENTION: irréversible)
    """
    try:
        pipeline = get_rag_pipeline()
        pipeline.clear()
        
        return {"message": "Vector store vidé avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du vidage: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Vérifie que le module RAG est opérationnel
    """
    try:
        pipeline = get_rag_pipeline()
        count = pipeline.count()
        
        return {
            "status": "healthy",
            "chunks_count": count,
            "ollama_url": pipeline.config.ollama_base_url,
            "embedding_model": pipeline.config.ollama_embedding_model if pipeline.config.embedding_provider == "ollama" else pipeline.config.openai_embedding_model,
            "llm_model": pipeline.config.ollama_llm_model if pipeline.config.llm_provider == "ollama" else pipeline.config.openai_llm_model
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG non disponible: {str(e)}")
