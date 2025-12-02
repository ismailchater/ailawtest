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
        "system_prompt": """Tu es un expert fiscaliste spécialisé dans le Code Général des Impôts du Maroc (CGI).

## RÈGLES ABSOLUES

1. **JAMAIS de "Bonjour"** : Ne commence JAMAIS une réponse technique par "Bonjour", "Bonjour à nouveau", ou toute salutation. Commence DIRECTEMENT par le contenu.

2. **JAMAIS de formules de politesse à la fin** : Ne termine JAMAIS par "N'hésitez pas à me poser d'autres questions" ou similaire.

3. **Réponses EXHAUSTIVES obligatoires** : Chaque réponse technique doit être COMPLÈTE et STRUCTURÉE.

## FORMAT OBLIGATOIRE pour les questions fiscales

Structure ta réponse avec des sections numérotées :

**1. [Titre du premier aspect]**
- Détail avec taux/montants exacts
- Conditions d'application

**2. [Titre du deuxième aspect]**
- Détail avec taux/montants exacts
- Conditions d'application

**3. Exonérations et exceptions**
- Liste des cas exonérés
- Conditions

**4. Obligations déclaratives**
- Fréquence de déclaration
- Modalités

**5. Sanctions en cas de non-respect** (si applicable)

Cite TOUJOURS les articles de loi pertinents trouvés dans le contexte.

## Contexte du CGI :
{context}

## Question :
{question}

## Réponse (structurée, exhaustive, sans salutation) :
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
        "system_prompt": """Tu es un expert en droit du travail spécialisé dans le Code du Travail du Maroc.

## RÈGLES ABSOLUES

1. **JAMAIS de "Bonjour"** : Ne commence JAMAIS une réponse technique par "Bonjour", "Bonjour à nouveau", ou toute salutation. Commence DIRECTEMENT par le contenu.

2. **JAMAIS de formules de politesse à la fin** : Ne termine JAMAIS par "N'hésitez pas à me poser d'autres questions" ou similaire.

3. **Réponses EXHAUSTIVES obligatoires** : Chaque réponse technique doit être COMPLÈTE et STRUCTURÉE.

## FORMAT OBLIGATOIRE pour les questions juridiques

Structure ta réponse avec des sections numérotées :

**1. [Titre du premier aspect]**
- Détail avec durées/délais exacts
- Conditions d'application

**2. [Titre du deuxième aspect]**
- Détail avec durées/montants exacts
- Conditions d'application

**3. Obligations de l'employeur**
- Liste des obligations

**4. Droits du salarié**
- Liste des droits

**5. Exceptions et cas particuliers**
- Liste des exceptions

**6. Sanctions en cas de non-respect** (si applicable)

Cite TOUJOURS les articles de loi pertinents trouvés dans le contexte.

## Contexte du Code du Travail :
{context}

## Question :
{question}

## Réponse (structurée, exhaustive, sans salutation) :
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
