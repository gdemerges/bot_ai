"""
Module de chunking pour le RAG
Supporte plusieurs stratégies de découpage de documents
"""
from typing import List, Optional
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """Représente un chunk de document"""
    content: str
    metadata: dict
    chunk_index: int
    start_char: int
    end_char: int


class DocumentChunker:
    """
    Découpe les documents en chunks avec différentes stratégies
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "recursive"  # "recursive", "sentence", "paragraph", "fixed"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # Séparateurs pour le découpage récursif (du plus spécifique au plus général)
        self.separators = [
            "\n\n",  # Paragraphes
            "\n",    # Lignes
            ". ",    # Phrases
            "? ",
            "! ",
            "; ",
            ", ",
            " ",     # Mots
            ""       # Caractères
        ]
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None
    ) -> List[Chunk]:
        """
        Découpe un texte en chunks selon la stratégie configurée
        """
        if metadata is None:
            metadata = {}
        
        if self.strategy == "recursive":
            return self._recursive_chunk(text, metadata)
        elif self.strategy == "sentence":
            return self._sentence_chunk(text, metadata)
        elif self.strategy == "paragraph":
            return self._paragraph_chunk(text, metadata)
        else:
            return self._fixed_chunk(text, metadata)
    
    def _recursive_chunk(
        self,
        text: str,
        metadata: dict,
        separators: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Découpage récursif avec les séparateurs
        """
        if separators is None:
            separators = self.separators
        
        chunks = []
        current_separators = list(separators)
        
        # Trouver le premier séparateur qui existe dans le texte
        separator = ""
        for sep in current_separators:
            if sep in text:
                separator = sep
                break
        
        if separator:
            splits = text.split(separator)
        else:
            splits = [text]
        
        current_chunk = ""
        current_start = 0
        
        for i, split in enumerate(splits):
            potential_chunk = current_chunk + (separator if current_chunk else "") + split
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Si le chunk actuel est assez grand, le sauvegarder
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        metadata=metadata.copy(),
                        chunk_index=len(chunks),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    ))
                    # Calculer l'overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    overlap_text = current_chunk[overlap_start:]
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + separator + split
                else:
                    # Le split lui-même est trop grand, essayer avec un séparateur plus petit
                    if len(current_separators) > 1:
                        sub_chunks = self._recursive_chunk(
                            split,
                            metadata,
                            current_separators[1:]
                        )
                        for sub_chunk in sub_chunks:
                            sub_chunk.chunk_index = len(chunks)
                            chunks.append(sub_chunk)
                    else:
                        # Découpage forcé par caractères
                        for j in range(0, len(split), self.chunk_size - self.chunk_overlap):
                            chunk_text = split[j:j + self.chunk_size]
                            chunks.append(Chunk(
                                content=chunk_text.strip(),
                                metadata=metadata.copy(),
                                chunk_index=len(chunks),
                                start_char=j,
                                end_char=j + len(chunk_text)
                            ))
                    current_chunk = ""
        
        # Ajouter le dernier chunk s'il reste du texte
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata=metadata.copy(),
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))
        
        return chunks
    
    def _sentence_chunk(self, text: str, metadata: dict) -> List[Chunk]:
        """
        Découpage par phrases
        """
        # Regex pour détecter les fins de phrases
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        metadata=metadata.copy(),
                        chunk_index=len(chunks),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    ))
                    current_start += len(current_chunk)
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata=metadata.copy(),
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))
        
        return chunks
    
    def _paragraph_chunk(self, text: str, metadata: dict) -> List[Chunk]:
        """
        Découpage par paragraphes
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_start = 0
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            
            if len(para) <= self.chunk_size:
                chunks.append(Chunk(
                    content=para,
                    metadata=metadata.copy(),
                    chunk_index=len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(para)
                ))
            else:
                # Paragraphe trop long, découper en phrases
                sub_chunks = self._sentence_chunk(para, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.start_char += current_start
                    sub_chunk.end_char += current_start
                    sub_chunk.chunk_index = len(chunks)
                    chunks.append(sub_chunk)
            
            current_start += len(para) + 2  # +2 pour \n\n
        
        return chunks
    
    def _fixed_chunk(self, text: str, metadata: dict) -> List[Chunk]:
        """
        Découpage fixe par nombre de caractères
        """
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            chunks.append(Chunk(
                content=chunk_text.strip(),
                metadata=metadata.copy(),
                chunk_index=len(chunks),
                start_char=i,
                end_char=i + len(chunk_text)
            ))
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[dict]
    ) -> List[Chunk]:
        """
        Découpe une liste de documents
        Chaque document doit avoir 'content' et optionnellement 'metadata'
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            if "source" in doc:
                metadata["source"] = doc["source"]
            
            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
