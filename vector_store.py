"""
Vector store module for Qdrant management.
Handles embedding storage and retrieval for any module.
Supports local Qdrant instance or Qdrant Cloud.
"""

import os
import hashlib
from typing import List, Optional, Dict, Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config import ModelConfig, QdrantConfig, get_openai_api_key


class VectorStoreManager:
    """
    Manages the Qdrant vector store for document embeddings.
    Supports multiple modules with separate collections.
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = ModelConfig.EMBEDDING_MODEL
    ):
        """
        Initialize the vector store manager.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: OpenAI embedding model to use
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._client: Optional[QdrantClient] = None
        self._vector_size = 1536  # text-embedding-3-small dimensions
    
    def _get_client(self) -> QdrantClient:
        """Get or create the Qdrant client."""
        if self._client is None:
            # Check for cloud configuration (env vars first, then Streamlit secrets)
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            # Try Streamlit secrets if not in env
            if not qdrant_url or not qdrant_api_key:
                try:
                    import streamlit as st
                    if hasattr(st, 'secrets'):
                        qdrant_url = st.secrets.get("QDRANT_URL", qdrant_url)
                        qdrant_api_key = st.secrets.get("QDRANT_API_KEY", qdrant_api_key)
                except Exception:
                    pass
            
            if qdrant_url and qdrant_api_key:
                # Qdrant Cloud
                self._client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
            else:
                # Local Qdrant
                self._client = QdrantClient(
                    host=QdrantConfig.HOST,
                    port=QdrantConfig.PORT
                )
        return self._client
    
    def _get_embeddings(self) -> OpenAIEmbeddings:
        """Get or create the embeddings instance."""
        if self._embeddings is None:
            api_key = get_openai_api_key()
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=api_key
            )
        return self._embeddings
    
    def _generate_doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document based on content hash."""
        content = f"{doc.metadata.get('file_name', '')}_{doc.metadata.get('page', '')}_{doc.metadata.get('chunk_id', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def collection_exists(self) -> bool:
        """Check if the collection already exists."""
        try:
            client = self._get_client()
            collections = client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False
    
    def create_collection(self):
        """Create a new collection if it doesn't exist."""
        client = self._get_client()
        
        if not self.collection_exists():
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE
                )
            )
    
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            batch_size: Number of documents to process at once
        """
        client = self._get_client()
        embeddings = self._get_embeddings()
        
        # Ensure collection exists
        self.create_collection()
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Generate embeddings
            texts = [doc.page_content for doc in batch]
            vectors = embeddings.embed_documents(texts)
            
            # Create points
            points = []
            for doc, vector in zip(batch, vectors):
                point_id = str(uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": doc.page_content,
                        "file_name": doc.metadata.get("file_name", "unknown"),
                        "page": doc.metadata.get("page", 0),
                        "chunk_id": doc.metadata.get("chunk_id", 0),
                        "source": doc.metadata.get("source", ""),
                        "module": doc.metadata.get("module", "")
                    }
                ))
            
            # Upsert points
            client.upsert(
                collection_name=self.collection_name,
                points=points
            )
    
    def delete_by_file(self, file_name: str):
        """Delete all vectors associated with a specific file."""
        client = self._get_client()
        
        client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_name",
                            match=models.MatchValue(value=file_name)
                        )
                    ]
                )
            )
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        client = self._get_client()
        
        if not self.collection_exists():
            return {"exists": False, "count": 0}
        
        info = client.get_collection(self.collection_name)
        return {
            "exists": True,
            "count": info.points_count
        }
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Perform similarity search on the vector store."""
        client = self._get_client()
        embeddings = self._get_embeddings()
        
        # Generate query embedding
        query_vector = embeddings.embed_query(query)
        
        # Search
        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )
        
        # Convert to Documents
        documents = []
        for result in results:
            doc = Document(
                page_content=result.payload.get("content", ""),
                metadata={
                    "file_name": result.payload.get("file_name", ""),
                    "page": result.payload.get("page", 0),
                    "chunk_id": result.payload.get("chunk_id", 0),
                    "source": result.payload.get("source", ""),
                    "module": result.payload.get("module", ""),
                    "score": result.score
                }
            )
            documents.append(doc)
        
        return documents
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Get a retriever-like object for compatibility."""
        k = 8
        if search_kwargs and "k" in search_kwargs:
            k = search_kwargs["k"]
        
        # Return a simple wrapper
        class QdrantRetriever:
            def __init__(self, manager, k):
                self.manager = manager
                self.k = k
            
            def invoke(self, query: str) -> List[Document]:
                return self.manager.similarity_search(query, k=self.k)
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.invoke(query)
        
        return QdrantRetriever(self, k)
    
    def clear_collection(self):
        """Delete all documents in the collection."""
        client = self._get_client()
        
        if self.collection_exists():
            client.delete_collection(self.collection_name)
            self.create_collection()


def create_vector_store_manager(module_config: Dict[str, Any]) -> VectorStoreManager:
    """
    Factory function to create a VectorStoreManager for a specific module.
    
    Args:
        module_config: Module configuration dictionary
        
    Returns:
        VectorStoreManager: Configured vector store manager instance
    """
    return VectorStoreManager(
        collection_name=module_config["collection_name"]
    )
