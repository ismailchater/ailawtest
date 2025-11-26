"""
RAG Chain module for question-answering logic.
Handles the retrieval-augmented generation pipeline.
"""

import re
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import ModelConfig, SYSTEM_PROMPT, get_openai_api_key
from vector_store import VectorStoreManager


# Patterns for conversational queries (non-CGI)
CONVERSATIONAL_PATTERNS = [
    r"^(salut|bonjour|bonsoir|hello|hi|hey|coucou)",
    r"^(ça va|comment vas|comment tu vas|tu vas bien|comment allez)",
    r"^(merci|thanks|thank you)",
    r"^(au revoir|bye|à bientôt|à plus)",
    r"^(qui es[ -]tu|tu es qui|c'est quoi|présente[ -]toi)",
    r"^(ok|d'accord|compris|super|parfait|génial|cool)$",
    r"^(oui|non|ouais|nope)$",
]


def is_conversational_query(question: str) -> bool:
    """
    Check if the question is a conversational query (greeting, thanks, etc.)
    
    Args:
        question: The user's question
        
    Returns:
        bool: True if it's a conversational query
    """
    question_lower = question.lower().strip()
    
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return True
    
    # Check if it's a very short non-fiscal query
    if len(question_lower) < 15 and not any(
        keyword in question_lower for keyword in [
            "impôt", "taxe", "tva", "is", "ir", "fiscal", 
            "taux", "article", "cgi", "déclar", "exonér",
            "société", "revenu", "bénéfice", "auto-entrepreneur"
        ]
    ):
        return True
    
    return False


class RAGChainBuilder:
    """
    Builds and manages the RAG (Retrieval Augmented Generation) chain.
    
    Combines retrieval from vector store with LLM generation.
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = ModelConfig.LLM_MODEL,
        temperature: float = ModelConfig.LLM_TEMPERATURE
    ):
        """
        Initialize the RAG chain builder.
        
        Args:
            vector_store_manager: Manager for the vector store
            model_name: Name of the OpenAI model to use
            temperature: Temperature setting for the LLM
        """
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None
        self._chain = None
        self._conversational_chain = None
    
    def _get_llm(self) -> ChatOpenAI:
        """
        Get or create the LLM instance.
        
        Returns:
            ChatOpenAI: Configured LLM instance
        """
        if self._llm is None:
            api_key = get_openai_api_key()
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=api_key
            )
        return self._llm
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for the RAG chain.
        
        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        return ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    def _create_conversational_prompt(self) -> ChatPromptTemplate:
        """
        Create the prompt template for conversational responses.
        
        Returns:
            ChatPromptTemplate: Configured conversational prompt
        """
        conversational_template = """Tu es un assistant fiscaliste marocain sympathique et professionnel.

L'utilisateur t'a envoyé un message conversationnel (salutation, remerciement, question générale).

Réponds de manière chaleureuse et naturelle en français. Si c'est une première interaction, présente-toi brièvement comme assistant spécialisé dans le Code Général des Impôts du Maroc et invite l'utilisateur à poser ses questions fiscales.

Message de l'utilisateur : {question}

Ta réponse chaleureuse :"""
        return ChatPromptTemplate.from_template(conversational_template)
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a single context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            str: Formatted context string
        """
        if not documents:
            return "Aucun contexte disponible."
        
        formatted_parts = []
        
        for i, doc in enumerate(documents, 1):
            page_num = doc.metadata.get("page", "N/A")
            content = doc.page_content.strip()
            formatted_parts.append(
                f"[Source {i} - Page {page_num}]\n{content}"
            )
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def build_chain(self):
        """
        Build the complete RAG chain.
        
        Returns:
            Runnable: The configured RAG chain
        """
        # Increase k to 8 for better coverage
        retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": 8}
        )
        prompt = self._create_prompt_template()
        llm = self._get_llm()
        
        self._chain = (
            {
                "context": retriever | self._format_documents,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return self._chain
    
    def _build_conversational_chain(self):
        """Build the conversational chain for non-CGI queries."""
        prompt = self._create_conversational_prompt()
        llm = self._get_llm()
        
        self._conversational_chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return self._conversational_chain
    
    def get_chain(self):
        """
        Get the RAG chain, building it if necessary.
        
        Returns:
            Runnable: The RAG chain
        """
        if self._chain is None:
            self.build_chain()
        return self._chain
    
    def get_conversational_chain(self):
        """
        Get the conversational chain, building it if necessary.
        
        Returns:
            Runnable: The conversational chain
        """
        if self._conversational_chain is None:
            self._build_conversational_chain()
        return self._conversational_chain
    
    def invoke(self, question: str) -> str:
        """
        Invoke the appropriate chain based on question type.
        
        Args:
            question: The user's question
            
        Returns:
            str: The generated answer
        """
        if is_conversational_query(question):
            chain = self.get_conversational_chain()
        else:
            chain = self.get_chain()
        
        return chain.invoke(question)
    
    def get_relevant_documents(self, question: str, k: int = 8) -> List[Document]:
        """
        Get relevant documents for a question without generating an answer.
        
        Args:
            question: The user's question
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: Relevant documents
        """
        return self.vector_store_manager.similarity_search(question, k=k)


class RAGQueryHandler:
    """
    High-level handler for RAG queries.
    
    Provides a simple interface for asking questions and getting answers.
    """
    
    def __init__(self, rag_chain: RAGChainBuilder):
        """
        Initialize the query handler.
        
        Args:
            rag_chain: The RAG chain builder to use
        """
        self.rag_chain = rag_chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a structured response.
        
        Args:
            question: The user's question
            
        Returns:
            Dict containing 'answer' and 'sources' keys
        """
        try:
            # Check if conversational
            is_conversational = is_conversational_query(question)
            
            # Get the answer
            answer = self.rag_chain.invoke(question)
            
            # Only get source documents for non-conversational queries
            if is_conversational:
                source_pages = []
            else:
                sources = self.rag_chain.get_relevant_documents(question)
                source_pages = list(set(
                    doc.metadata.get("page", "N/A") 
                    for doc in sources
                ))
            
            return {
                "answer": answer,
                "sources": source_pages,
                "success": True,
                "error": None,
                "is_conversational": is_conversational
            }
            
        except Exception as e:
            return {
                "answer": None,
                "sources": [],
                "success": False,
                "error": str(e),
                "is_conversational": False
            }


def create_rag_chain(
    vector_store_manager: VectorStoreManager
) -> RAGChainBuilder:
    """
    Factory function to create a RAG chain builder.
    
    Args:
        vector_store_manager: The vector store manager to use
        
    Returns:
        RAGChainBuilder: Configured RAG chain builder
    """
    return RAGChainBuilder(vector_store_manager)
