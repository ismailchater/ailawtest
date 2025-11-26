"""
Vector store module for ChromaDB management.
Handles embedding storage and retrieval operations.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import ModelConfig, VectorStoreConfig, get_openai_api_key


class VectorStoreManager:
    """
    Manages the ChromaDB vector store for document embeddings.
    
    Handles creation, persistence, and retrieval from the vector store.
    """
    
    def __init__(
        self,
        persist_directory: str = VectorStoreConfig.PERSIST_DIRECTORY,
        collection_name: str = VectorStoreConfig.COLLECTION_NAME,
        embedding_model: str = ModelConfig.EMBEDDING_MODEL
    ):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vector_store: Optional[Chroma] = None
    
    def _get_embeddings(self) -> OpenAIEmbeddings:
        """
        Get or create the embeddings instance.
        
        Returns:
            OpenAIEmbeddings: Configured embeddings instance
        """
        if self._embeddings is None:
            api_key = get_openai_api_key()
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=api_key
            )
        return self._embeddings
    
    def vector_store_exists(self) -> bool:
        """
        Check if a persisted vector store already exists.
        
        Returns:
            bool: True if vector store exists on disk
        """
        chroma_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        return os.path.exists(chroma_path)
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of documents to embed and store
            
        Returns:
            Chroma: The created vector store
        """
        embeddings = self._get_embeddings()
        
        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        return self._vector_store
    
    def load_vector_store(self) -> Chroma:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma: The loaded vector store
        """
        embeddings = self._get_embeddings()
        
        self._vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            collection_name=self.collection_name
        )
        
        return self._vector_store
    
    def get_or_create_vector_store(
        self,
        documents: Optional[List[Document]] = None
    ) -> Chroma:
        """
        Get existing vector store or create a new one.
        
        Args:
            documents: Documents to use if creating new store
            
        Returns:
            Chroma: The vector store instance
            
        Raises:
            ValueError: If no existing store and no documents provided
        """
        if self._vector_store is not None:
            return self._vector_store
        
        if self.vector_store_exists():
            return self.load_vector_store()
        
        if documents is None:
            raise ValueError(
                "Aucune base vectorielle existante et aucun document fourni. "
                "Veuillez fournir des documents pour créer la base."
            )
        
        return self.create_vector_store(documents)
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever from the vector store.
        
        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 4})
            
        Returns:
            VectorStoreRetriever: Configured retriever
        """
        if self._vector_store is None:
            raise ValueError(
                "Vector store not initialized. "
                "Call get_or_create_vector_store first."
            )
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self._vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Document]: Most similar documents
        """
        if self._vector_store is None:
            raise ValueError(
                "Vector store not initialized. "
                "Call get_or_create_vector_store first."
            )
        
        return self._vector_store.similarity_search(query, k=k)


def get_vector_store_manager() -> VectorStoreManager:
    """
    Factory function to create a VectorStoreManager.
    
    Returns:
        VectorStoreManager: Configured vector store manager instance
    """
    return VectorStoreManager()

