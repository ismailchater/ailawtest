"""
Document loader module for PDF processing.
Handles loading and chunking of PDF documents.
"""

import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import ChunkingConfig, DocumentConfig


class PDFLoadError(Exception):
    """Custom exception for PDF loading errors."""
    pass


class DocumentProcessor:
    """
    Handles PDF document loading and text chunking.
    
    Attributes:
        pdf_path: Path to the PDF file
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between consecutive chunks
    """
    
    def __init__(
        self,
        pdf_path: str = DocumentConfig.PDF_PATH,
        chunk_size: int = ChunkingConfig.CHUNK_SIZE,
        chunk_overlap: int = ChunkingConfig.CHUNK_OVERLAP
    ):
        """
        Initialize the document processor.
        
        Args:
            pdf_path: Path to the PDF file to process
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_splitter = self._create_text_splitter()
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create and configure the text splitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def validate_pdf_exists(self) -> bool:
        """
        Check if the PDF file exists.
        
        Returns:
            bool: True if file exists, False otherwise
        """
        return os.path.exists(self.pdf_path)
    
    def load_pdf(self) -> List[Document]:
        """
        Load the PDF document.
        
        Returns:
            List[Document]: List of document pages
            
        Raises:
            PDFLoadError: If the PDF file cannot be loaded
        """
        if not self.validate_pdf_exists():
            raise PDFLoadError(
                f"Le fichier PDF '{self.pdf_path}' n'a pas été trouvé. "
                "Veuillez placer le fichier 'cgi_maroc.pdf' dans le répertoire racine."
            )
        
        try:
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            if not documents:
                raise PDFLoadError(
                    f"Le fichier PDF '{self.pdf_path}' est vide ou ne peut pas être lu."
                )
            
            return documents
            
        except PDFLoadError:
            raise
        except Exception as e:
            raise PDFLoadError(
                f"Erreur lors du chargement du PDF: {str(e)}"
            )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List[Document]: List of chunked documents
        """
        chunks = self._text_splitter.split_documents(documents)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source"] = self.pdf_path
        
        return chunks
    
    def load_and_split(self) -> List[Document]:
        """
        Load PDF and split into chunks in one operation.
        
        Returns:
            List[Document]: List of chunked documents
            
        Raises:
            PDFLoadError: If loading or processing fails
        """
        documents = self.load_pdf()
        return self.split_documents(documents)


def get_document_processor(
    pdf_path: str = DocumentConfig.PDF_PATH
) -> DocumentProcessor:
    """
    Factory function to create a DocumentProcessor.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        DocumentProcessor: Configured document processor instance
    """
    return DocumentProcessor(pdf_path=pdf_path)

