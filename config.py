"""
Configuration module for the multi-module legal assistant application.
Supports multiple legal documents: CGI (taxes), Code du Travail, etc.
Uses Qdrant for vector storage with folder-based document management.
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
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.2


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for Qdrant vector database."""
    HOST: str = "localhost"
    PORT: int = 6333
    # For cloud: set QDRANT_URL and QDRANT_API_KEY in environment


# Base folder for all document modules
DOCUMENTS_BASE_FOLDER = "./documents"


# =============================================================================
# MODULE CONFIGURATIONS
# =============================================================================

MODULES: Dict[str, Dict[str, Any]] = {
    "cgi": {
  "id": "cgi",
  "name": "Loi n° 17-95 - Sociétés Anonymes",
  "short_name": "Loi SA 17-95",
  "description": "Droit des sociétés anonymes au Maroc : constitution, administration, assemblées, capital et sanctions.",
  "documents_folder": "./documents/cgi", 
  "collection_name": "Sociétés Anonymes_maroc",
  "icon": "🏢",
  "color": "#1E88E5",
  "enabled": True,
        "system_prompt": """Tu es un expert juriste marocain spécialisé en droit des sociétés, et plus particulièrement sur la Loi n° 17-95 relative aux sociétés anonymes.

INSTRUCTIONS :
1. Base ta réponse sur le CONTEXTE ci-dessous
2. Cite les sources avec le nom du fichier et la page : "[Fichier: X, Page Y]"
3. Sois COMPLET : cite TOUS les éléments des listes (si 6 points, cite les 6)
4. Si le contexte contient des infos pertinentes, utilise-les même si pas exactement la question posée
5. Dis "Le contexte ne contient pas cette information spécifique" SEULEMENT si vraiment rien de pertinent
6. Pas de salutations, commence directement par la réponse

CONTEXTE DU SA :
{context}

QUESTION : {question}

RÉPONSE :"""
    },
    "cdt": {
        "id": "cdt",
        "name": "Code du Travail",
        "short_name": "CDT",
        "description": "Droit du travail marocain, contrats, licenciement, congés",
        "documents_folder": "./documents/cdt",  # Folder with multiple PDFs
        "collection_name": "cdt_maroc",
        "icon": "👷",
        "color": "#8B7355",
        "enabled": True,
        "system_prompt": """Tu es un expert en droit du travail marocain. Réponds à partir du CONTEXTE fourni.

INSTRUCTIONS :
1. Base ta réponse sur le CONTEXTE ci-dessous
2. Cite les sources avec le nom du fichier et la page : "[Fichier: X, Page Y]"
3. Sois COMPLET : cite TOUS les éléments des listes (si 6 points, cite les 6)
4. Si le contexte contient des infos pertinentes, utilise-les même si pas exactement la question posée
5. Dis "Le contexte ne contient pas cette information spécifique" SEULEMENT si vraiment rien de pertinent
6. Pas de salutations, commence directement par la réponse

CONTEXTE DU CODE DU TRAVAIL :
{context}

QUESTION : {question}

RÉPONSE :"""
    },
    "famille": {
        "id": "famille",
        "name": "Code de la Famille",
        "short_name": "Famille",
        "description": "Mariage, divorce, filiation, héritage, tutelle",
        "documents_folder": "./documents/famille",
        "collection_name": "famille_maroc",
        "icon": "👨‍👩‍👧‍👦",
        "color": "#9B6B8C",
        "enabled": True,
        "system_prompt": """Tu es un expert en droit de la famille marocain. Réponds à partir du CONTEXTE fourni.

INSTRUCTIONS :
1. Base ta réponse sur le CONTEXTE ci-dessous
2. Cite les sources avec le nom du fichier et la page : "[Fichier: X, Page Y]"
3. Sois COMPLET : cite TOUS les éléments des listes (si 6 points, cite les 6)
4. Si le contexte contient des infos pertinentes, utilise-les même si pas exactement la question posée
5. Dis "Le contexte ne contient pas cette information spécifique" SEULEMENT si vraiment rien de pertinent
6. Pas de salutations, commence directement par la réponse

CONTEXTE DU CODE DE LA FAMILLE :
{context}

QUESTION : {question}

RÉPONSE :"""
    },
    "immobilier": {
        "id": "immobilier",
        "name": "Immobilier",
        "short_name": "Immo",
        "description": "Droit immobilier, propriété, location, copropriété",
        "documents_folder": "./documents/immobilier",
        "collection_name": "immobilier_maroc",
        "icon": "🏠",
        "color": "#5D8AA8",
        "enabled": False,  # Coming soon
        "system_prompt": ""
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
