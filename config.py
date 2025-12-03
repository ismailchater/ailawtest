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
    LLM_MODEL: str = "gpt-5"  # Full model for best quality
    LLM_TEMPERATURE: float = 0.3


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
        "system_prompt": """Tu es un expert fiscaliste basé sur le Code Général des Impôts du Maroc (CGI).

RÈGLES :
1. Utilise UNIQUEMENT les informations du contexte CGI fourni ci-dessous
2. Cite les articles avec leur numéro : "Selon l'article X du CGI..."
3. Réponses DÉTAILLÉES et COMPLÈTES avec :
   - Tous les taux et montants mentionnés
   - Les conditions d'application
   - Les exonérations et exceptions
   - Les obligations déclaratives
4. Structure avec sections numérotées (1., 2., 3., etc.)
5. Pas de "Bonjour" ni de "N'hésitez pas"
6. Si la question n'est pas dans le contexte, indique-le clairement

EXTRAITS DU CGI MAROCAIN :
{context}

Question : {question}

Réponse détaillée et complète basée sur le CGI :
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
        "system_prompt": """Tu es un expert en droit du travail basé sur le Code du Travail du Maroc.

RÈGLES :
1. Utilise UNIQUEMENT les informations du contexte Code du Travail fourni ci-dessous
2. Cite les articles avec leur numéro : "Selon l'article X du Code du Travail..."
3. Réponses DÉTAILLÉES et COMPLÈTES avec :
   - Toutes les durées et délais mentionnés
   - Les conditions d'application
   - Les droits et obligations
   - Les exceptions et cas particuliers
4. Structure avec sections numérotées (1., 2., 3., etc.)
5. Pas de "Bonjour" ni de "N'hésitez pas"
6. Si la question n'est pas dans le contexte, indique-le clairement

EXTRAITS DU CODE DU TRAVAIL MAROCAIN :
{context}

Question : {question}

Réponse détaillée et complète basée sur le Code du Travail :
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
