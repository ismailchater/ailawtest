"""
RAG Chain module for question-answering logic.
Handles the retrieval-augmented generation pipeline for any module.
"""

import re
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import ModelConfig, get_openai_api_key
from vector_store import VectorStoreManager


# Patterns for conversational queries (non-legal)
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
    """Check if the question is a conversational query."""
    question_lower = question.lower().strip()
    
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return True
    
    # Check for short non-legal queries
    legal_keywords = [
        "impôt", "taxe", "tva", "is", "ir", "fiscal", "taux", "article",
        "cgi", "déclar", "exonér", "société", "revenu", "bénéfice",
        "auto-entrepreneur", "travail", "contrat", "licenciement", "congé",
        "salaire", "employeur", "salarié", "cdd", "cdi", "préavis",
        "indemnité", "syndicat", "grève", "heures", "smig"
    ]
    
    if len(question_lower) < 15 and not any(kw in question_lower for kw in legal_keywords):
        return True
    
    return False


class RAGChainBuilder:
    """
    Builds and manages the RAG chain for any legal module.
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        system_prompt: str,
        module_name: str,
        model_name: str = ModelConfig.LLM_MODEL,
        temperature: float = ModelConfig.LLM_TEMPERATURE
    ):
        """
        Initialize the RAG chain builder.
        
        Args:
            vector_store_manager: Manager for the vector store
            system_prompt: The system prompt for this module
            module_name: Name of the module for conversational responses
            model_name: Name of the OpenAI model to use
            temperature: Temperature setting for the LLM
        """
        self.vector_store_manager = vector_store_manager
        self.system_prompt = system_prompt
        self.module_name = module_name
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None
        self._chain = None
        self._conversational_chain = None
    
    def _get_llm(self) -> ChatOpenAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            api_key = get_openai_api_key()
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=api_key
            )
        return self._llm
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for the RAG chain."""
        return ChatPromptTemplate.from_template(self.system_prompt)
    
    def _create_conversational_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for conversational responses."""
        conversational_template = f"""Tu es un assistant juridique marocain sympathique et professionnel, spécialisé dans le {self.module_name}.

L'utilisateur t'a envoyé un message conversationnel (salutation, remerciement, question générale).

Réponds de manière chaleureuse et naturelle en français. Si c'est une première interaction, présente-toi brièvement comme assistant spécialisé dans le {self.module_name} et invite l'utilisateur à poser ses questions.

Message de l'utilisateur : {{question}}

Ta réponse chaleureuse :"""
        return ChatPromptTemplate.from_template(conversational_template)
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format retrieved documents into a single context string."""
        if not documents:
            return "Aucun contexte disponible."
        
        formatted_parts = []
        for i, doc in enumerate(documents, 1):
            page_num = doc.metadata.get("page", "N/A")
            content = doc.page_content.strip()
            formatted_parts.append(f"[Source {i} - Page {page_num}]\n{content}")
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def build_chain(self):
        """Build the complete RAG chain."""
        retriever = self.vector_store_manager.get_retriever(search_kwargs={"k": 8})
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
        """Build the conversational chain for non-legal queries."""
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
        """Get the RAG chain, building it if necessary."""
        if self._chain is None:
            self.build_chain()
        return self._chain
    
    def get_conversational_chain(self):
        """Get the conversational chain, building it if necessary."""
        if self._conversational_chain is None:
            self._build_conversational_chain()
        return self._conversational_chain
    
    def invoke(self, question: str) -> str:
        """Invoke the appropriate chain based on question type."""
        if is_conversational_query(question):
            chain = self.get_conversational_chain()
        else:
            chain = self.get_chain()
        
        return chain.invoke(question)
    
    def get_relevant_documents(self, question: str, k: int = 8) -> List[Document]:
        """Get relevant documents for a question."""
        return self.vector_store_manager.similarity_search(question, k=k)


class RAGQueryHandler:
    """High-level handler for RAG queries."""
    
    def __init__(self, rag_chain: RAGChainBuilder):
        """Initialize the query handler."""
        self.rag_chain = rag_chain
    
    def _format_conversation_history(self, history: List[Dict[str, str]], max_exchanges: int = 5) -> str:
        """
        Format the last N exchanges from conversation history.
        
        Args:
            history: List of message dictionaries with 'role' and 'content' keys
            max_exchanges: Maximum number of exchanges (user+assistant pairs) to include
            
        Returns:
            Formatted conversation history string
        """
        if not history or len(history) <= 1:
            return ""
        
        # Skip the initial assistant greeting, take last messages
        # Each exchange = 2 messages (user + assistant)
        relevant_history = history[1:]  # Skip first greeting
        max_messages = max_exchanges * 2
        
        if len(relevant_history) > max_messages:
            relevant_history = relevant_history[-max_messages:]
        
        if not relevant_history:
            return ""
        
        formatted_parts = []
        for msg in relevant_history:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            # Truncate long messages for context
            content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
            formatted_parts.append(f"{role}: {content}")
        
        return "\n".join(formatted_parts)
    
    def ask(self, question: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Ask a question and get a structured response.
        
        Args:
            question: The user's question
            conversation_history: Optional list of previous messages for context
        """
        try:
            is_conversational = is_conversational_query(question)
            
            # Build question with conversation context
            if conversation_history and len(conversation_history) > 1:
                history_text = self._format_conversation_history(conversation_history)
                if history_text:
                    question_with_context = f"""Historique de la conversation récente:
{history_text}

Nouvelle question de l'utilisateur: {question}"""
                else:
                    question_with_context = question
            else:
                question_with_context = question
            
            answer = self.rag_chain.invoke(question_with_context)
            
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
    vector_store_manager: VectorStoreManager,
    module_config: Dict[str, Any]
) -> RAGChainBuilder:
    """
    Factory function to create a RAG chain builder for a specific module.
    
    Args:
        vector_store_manager: The vector store manager to use
        module_config: Module configuration dictionary
        
    Returns:
        RAGChainBuilder: Configured RAG chain builder
    """
    return RAGChainBuilder(
        vector_store_manager=vector_store_manager,
        system_prompt=module_config["system_prompt"],
        module_name=module_config["name"]
    )
