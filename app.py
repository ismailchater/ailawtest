"""
Main Streamlit application for the RAG Tax Bot.
Provides a chat interface for querying the Moroccan Tax Code (CGI).
"""

import streamlit as st
from typing import Optional, Tuple

from document_loader import DocumentProcessor, PDFLoadError
from vector_store import VectorStoreManager
from rag_chain import RAGChainBuilder, RAGQueryHandler


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="CGI Maroc - Assistant Fiscal",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# Styling
# =============================================================================

def apply_custom_styles():
    """Apply custom CSS styles to the application."""
    st.markdown("""
        <style>
        /* Main container styling */
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        
        /* Chat message styling */
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* Title styling */
        .main-title {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #e94560 0%, #ff6b6b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-family: 'Segoe UI', sans-serif;
            color: #a0a0a0;
            text-align: center;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        
        /* Success/Error messages */
        .status-success {
            background-color: rgba(46, 204, 113, 0.1);
            border-left: 4px solid #2ecc71;
            padding: 1rem;
            border-radius: 4px;
        }
        
        .status-error {
            background-color: rgba(231, 76, 60, 0.1);
            border-left: 4px solid #e74c3c;
            padding: 1rem;
            border-radius: 4px;
        }
        
        /* Sidebar styling */
        .sidebar-info {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Cached Resource Loading
# =============================================================================

@st.cache_resource(show_spinner=False)
def initialize_document_processor() -> DocumentProcessor:
    """
    Initialize and cache the document processor.
    
    Returns:
        DocumentProcessor: Cached document processor instance
    """
    return DocumentProcessor()


@st.cache_resource(show_spinner=False)
def initialize_vector_store_manager() -> VectorStoreManager:
    """
    Initialize and cache the vector store manager.
    
    Returns:
        VectorStoreManager: Cached vector store manager instance
    """
    return VectorStoreManager()


@st.cache_resource(show_spinner=False)
def load_and_index_documents(
    _doc_processor: DocumentProcessor,
    _vs_manager: VectorStoreManager
) -> Tuple[bool, Optional[str], int]:
    """
    Load PDF documents and create/load vector store.
    
    Args:
        _doc_processor: Document processor instance (underscore prefix for cache)
        _vs_manager: Vector store manager instance
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str], num_chunks: int)
    """
    try:
        # Check if vector store already exists
        if _vs_manager.vector_store_exists():
            _vs_manager.load_vector_store()
            return True, None, -1  # -1 indicates loaded from cache
        
        # Load and process documents
        documents = _doc_processor.load_and_split()
        
        # Create vector store
        _vs_manager.create_vector_store(documents)
        
        return True, None, len(documents)
        
    except PDFLoadError as e:
        return False, str(e), 0
    except Exception as e:
        return False, f"Erreur inattendue: {str(e)}", 0


@st.cache_resource(show_spinner=False)
def initialize_rag_chain(
    _vs_manager: VectorStoreManager
) -> RAGChainBuilder:
    """
    Initialize and cache the RAG chain.
    
    Args:
        _vs_manager: Vector store manager instance
        
    Returns:
        RAGChainBuilder: Cached RAG chain builder
    """
    return RAGChainBuilder(_vs_manager)


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-title">📚 Assistant CGI Maroc</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Expert fiscaliste virtuel spécialisé dans le Code Général des Impôts</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the sidebar with application info."""
    with st.sidebar:
        st.header("ℹ️ À propos")
        st.markdown("""
        Cet assistant vous aide à naviguer dans le **Code Général des Impôts du Maroc**.
        
        **Fonctionnalités :**
        - 🔍 Recherche sémantique dans le CGI
        - 📖 Citations des articles pertinents
        - 💬 Interface de chat intuitive
        
        **Comment utiliser :**
        1. Posez votre question fiscale
        2. L'assistant recherche dans le CGI
        3. Recevez une réponse avec les articles cités
        """)
        
        st.divider()
        
        st.markdown("**Exemples de questions :**")
        example_questions = [
            "Quel est le taux d'IS applicable aux sociétés ?",
            "Quelles sont les exonérations de TVA ?",
            "Comment calculer l'IR sur les salaires ?",
            "Quelles sont les obligations déclaratives des entreprises ?"
        ]
        for question in example_questions:
            st.markdown(f"- _{question}_")
        
        st.divider()
        st.caption("Propulsé par OpenAI GPT-4o et LangChain")


def render_initialization_status(success: bool, error: Optional[str], num_chunks: int):
    """
    Render the initialization status message.
    
    Args:
        success: Whether initialization was successful
        error: Error message if any
        num_chunks: Number of document chunks created (-1 if loaded from cache)
    """
    if success:
        if num_chunks == -1:
            st.success("✅ Base de connaissances chargée depuis le cache")
        else:
            st.success(f"✅ Base de connaissances créée ({num_chunks} segments indexés)")
    else:
        st.error(f"❌ {error}")
        st.info(
            "💡 Assurez-vous que le fichier 'cgi_maroc.pdf' est présent "
            "dans le répertoire de l'application."
        )


def initialize_chat_history():
    """Initialize the chat history in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Bonjour ! Je suis votre assistant spécialisé dans le "
                    "Code Général des Impôts du Maroc. 🇲🇦\n\n"
                    "Posez-moi vos questions fiscales et je vous répondrai "
                    "en citant les articles pertinents du CGI."
                )
            }
        ]


def render_chat_history():
    """Render the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(query_handler: RAGQueryHandler):
    """
    Handle user input and generate responses.
    
    Args:
        query_handler: The RAG query handler for processing questions
    """
    if prompt := st.chat_input("Posez votre question sur le CGI..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            # Use different spinner text based on query type
            with st.spinner("Réflexion en cours..."):
                result = query_handler.ask(prompt)
                
                if result["success"]:
                    response = result["answer"]
                    
                    # Only add source pages for non-conversational queries
                    is_conversational = result.get("is_conversational", False)
                    if not is_conversational and result["sources"]:
                        pages = [str(p) for p in result["sources"] if p != "N/A"]
                        if pages:
                            # Sort pages numerically
                            sorted_pages = sorted(set(pages), key=lambda x: int(x) if x.isdigit() else 0)
                            response += f"\n\n📄 _Sources: Pages {', '.join(sorted_pages)}_"
                else:
                    response = (
                        f"⚠️ Une erreur s'est produite: {result['error']}\n\n"
                        "Veuillez réessayer ou reformuler votre question."
                    )
                
                st.markdown(response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})


def render_clear_chat_button():
    """Render a button to clear chat history."""
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Nouvelle conversation", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Apply custom styles
    apply_custom_styles()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Initialize components
    doc_processor = initialize_document_processor()
    vs_manager = initialize_vector_store_manager()
    
    # Load and index documents
    with st.spinner("🔄 Chargement de la base de connaissances..."):
        success, error, num_chunks = load_and_index_documents(doc_processor, vs_manager)
    
    # Show initialization status
    render_initialization_status(success, error, num_chunks)
    
    if not success:
        st.stop()
    
    # Initialize RAG chain
    rag_chain = initialize_rag_chain(vs_manager)
    query_handler = RAGQueryHandler(rag_chain)
    
    st.divider()
    
    # Initialize chat history
    initialize_chat_history()
    
    # Render chat history
    render_chat_history()
    
    # Handle user input
    handle_user_input(query_handler)
    
    # Render clear chat button
    st.divider()
    render_clear_chat_button()


if __name__ == "__main__":
    main()

