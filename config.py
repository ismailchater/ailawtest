"""
Configuration module for the multi-module legal assistant application.
Supports multiple legal documents: CGI (taxes), Code du Travail, etc.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for document chunking."""
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for OpenAI models."""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-5-mini"
    LLM_TEMPERATURE: float = 0.5


# =============================================================================
# MODULE CONFIGURATIONS
# =============================================================================

MODULES: Dict[str, Dict[str, Any]] = {
    "cgi": {
        "id": "cgi",
        "name": "Code Général des Impôts",
        "short_name": "CGI",
        "description": "Fiscalité marocaine, IS, IR, TVA, taxes et impôts",
        "pdf_path": "cgi_maroc.pdf",
        "persist_directory": "./chroma_db_cgi",
        "collection_name": "cgi_maroc_docs",
        "icon": "💰",
        "color": "#D4A574",
        "system_prompt": """Tu es un expert fiscaliste marocain spécialisé dans le CGI.

RÈGLES :
- Pas de "Bonjour" ni de "N'hésitez pas" 
- Réponses complètes et structurées avec sections numérotées
- Cite les articles de loi du contexte
- Si la question n'est pas fiscale, explique poliment que tu es spécialisé en fiscalité marocaine

Contexte CGI :
{context}

Question : {question}
"""
    },
    "cdt": {
        "id": "cdt",
        "name": "Code du Travail",
        "short_name": "CDT",
        "description": "Droit du travail marocain, contrats, licenciement, congés",
        "pdf_path": "cdt_maroc.pdf",
        "persist_directory": "./chroma_db_cdt",
        "collection_name": "cdt_maroc_docs",
        "icon": "👷",
        "color": "#8B7355",
        "system_prompt": """Tu es un expert en droit du travail marocain spécialisé dans le Code du Travail.

RÈGLES :
- Pas de "Bonjour" ni de "N'hésitez pas"
- Réponses complètes et structurées avec sections numérotées
- Cite les articles de loi du contexte
- Si la question n'est pas sur le droit du travail, explique poliment que tu es spécialisé en droit du travail marocain

Contexte Code du Travail :
{context}

Question : {question}
"""
    }
}


def get_module_config(module_id: str) -> Dict[str, Any]:
    """
    Get configuration for a specific module.
    
    Args:
        module_id: The module identifier (e.g., 'cgi', 'cdt')
        
    Returns:
        Dict containing module configuration
        
    Raises:
        ValueError: If module_id is not found
    """
    if module_id not in MODULES:
        raise ValueError(f"Module '{module_id}' not found. Available: {list(MODULES.keys())}")
    return MODULES[module_id]


def get_openai_api_key() -> str:
    """
    Retrieve OpenAI API key from environment or Streamlit secrets.
    
    Returns:
        str: The OpenAI API key
        
    Raises:
        ValueError: If no API key is found
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        return api_key
    
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except Exception:
        pass
    
    raise ValueError(
        "Clé API OpenAI non trouvée. "
        "Définissez OPENAI_API_KEY dans les variables d'environnement "
        "ou dans .streamlit/secrets.toml"
    )
