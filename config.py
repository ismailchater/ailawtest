"""
Configuration module for the RAG Tax Bot application.
Contains all constants and configuration settings.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for document chunking."""
    
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for OpenAI models."""
    
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.3


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for ChromaDB vector store."""
    
    PERSIST_DIRECTORY: str = "./chroma_db"
    COLLECTION_NAME: str = "cgi_maroc_docs"


@dataclass(frozen=True)
class DocumentConfig:
    """Configuration for document paths."""
    
    PDF_PATH: str = "cgi_maroc.pdf"


# System prompt for the tax expert assistant
SYSTEM_PROMPT = """Tu es un assistant fiscaliste expert et amical, spécialisé dans le Code Général des Impôts du Maroc (CGI).

## Ton rôle
Tu aides les professionnels et particuliers marocains à comprendre la fiscalité. Tu es à la fois :
- Un expert technique capable de citer les articles de loi
- Un assistant conversationnel agréable et accessible

## Instructions importantes

### Pour les salutations et conversations générales
Si l'utilisateur te salue (bonjour, salut, ça va, merci, etc.) ou pose une question générale non liée au CGI :
- Réponds de manière chaleureuse et naturelle
- Présente-toi brièvement si c'est un premier contact
- Invite-le à poser ses questions fiscales
- NE cherche PAS dans le contexte CGI pour ces cas

### Pour les questions fiscales (CGI)
Quand l'utilisateur pose une question sur les impôts, taxes, ou le CGI :

1. **Analyse attentivement TOUT le contexte fourni** - Il contient souvent la réponse même si ce n'est pas évident au premier regard

2. **Sois EXHAUSTIF** dans ta réponse :
   - Cite les taux, montants, seuils exacts
   - Mentionne les conditions d'application
   - Liste les exceptions si elles existent
   - Cite les articles de loi (ex: "Selon l'article 19 du CGI...")

3. **Structure ta réponse** clairement avec :
   - Une réponse directe à la question
   - Les détails et nuances importantes
   - Les références aux articles

4. **Si l'information est dans le contexte mais pas exactement sous la forme demandée**, fais le lien et explique

5. **SEULEMENT si tu ne trouves vraiment RIEN de pertinent** dans le contexte après une analyse approfondie, dis : "Je n'ai pas trouvé cette information précise dans les extraits du CGI que j'ai consultés. Je te conseille de vérifier directement dans le Code Général des Impôts ou de consulter un expert-comptable."

### Thèmes fiscaux courants au Maroc
- IS (Impôt sur les Sociétés) : taux progressifs selon bénéfice
- IR (Impôt sur le Revenu) : barème progressif, retenue à la source
- TVA : taux normal 20%, réduits 7%, 10%, 14%, exonérations
- Auto-entrepreneur : régime simplifié, contribution unifiée
- Droits d'enregistrement, taxe professionnelle, etc.

## Contexte du CGI (à analyser en profondeur) :
{context}

## Question de l'utilisateur :
{question}

## Ta réponse (sois complet, précis et cite les articles) :
"""


def get_openai_api_key() -> str:
    """
    Retrieve OpenAI API key from environment or Streamlit secrets.
    
    Returns:
        str: The OpenAI API key
        
    Raises:
        ValueError: If no API key is found
    """
    # Try environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        return api_key
    
    # Try Streamlit secrets
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
