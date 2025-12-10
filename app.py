"""
IYYA - Assistant Juridique Marocain
Multi-module legal assistant for Moroccan law (CGI, Code du Travail, etc.)
"""

import streamlit as st
from typing import Optional, Tuple, Dict, Any

from config import MODULES, get_module_config
from document_loader import FolderDocumentProcessor, create_folder_processor, PDFLoadError
from vector_store import VectorStoreManager, create_vector_store_manager
from rag_chain import RAGChainBuilder, RAGQueryHandler, create_rag_chain


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="IYYA - Assistant Juridique",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# Golden Theme Styling (Updated to match Photo)
# =============================================================================

def apply_golden_theme():
    """Apply the golden/beige IYYA theme matching the reference image."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
        
        /* Main background - Beige/Tan solid color like the image background */
        .stApp {
            background-color: #D3C7B4;
            background-image: linear-gradient(180deg, #DBCFB9 0%, #CCBFAB 100%);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main title styling */
        .main-title {
            font-family: 'Playfair Display', serif;
            font-size: 3rem;
            font-weight: 700;
            color: #8B6914;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.3);
        }
        
        /* "Choisissez votre module" Header */
        .module-header {
            font-family: 'Playfair Display', serif;
            font-size: 2.2rem;
            font-weight: 700;
            color: #9E7E38; /* Golden Brown */
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
        }
        
        /* The horizontal line below the header */
        .header-line {
            border: 0;
            height: 2px;
            background-image: linear-gradient(to right, transparent, #B89656, transparent);
            margin-bottom: 3rem;
        }

        /* --- CARD STYLING --- */
        
        /* The container with border (The White Card) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FDF9F3 !important; /* Light Cream */
            border: 1px solid #EBE3D3 !important;
            border-radius: 15px !important;
            padding: 2rem !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        
        /* Hover effect on card */
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.08);
        }

        /* Card Icon */
        .card-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }

        /* Card Title */
        .card-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: #594A35;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }

        /* Card Description */
        .card-desc {
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            color: #7A6F60;
            line-height: 1.5;
            margin-bottom: 1rem;
        }

        /* --- BUTTON STYLING --- */
        
        /* Primary Button (The Golden Button below the card) */
        .stButton > button {
            background-color: #C69346 !important; /* Ochre Gold */
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            font-weight: 500;
            width: 100%;
            margin-top: 10px;
            box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            transition: background-color 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #B08036 !important; /* Darker Gold */
            box-shadow: 0 5px 10px rgba(0,0,0,0.15);
        }
        
        .stButton > button:disabled {
            background-color: #D6Cebb !important;
            color: #999 !important;
        }
        
        /* Chat Interface Styling */
        .stChatMessage {
            background-color: #FDF9F3 !important;
            border: 1px solid #E0D5C0 !important;
            border-radius: 12px;
        }
        
        .stChatInput > div {
            background-color: #FDF9F3 !important;
            border: 2px solid #C69346 !important;
        }

        /* Powered by */
        .powered-by {
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            color: #7A6B5A;
            text-align: center;
            margin-top: 4rem;
            opacity: 0.7;
        }
        </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Session State Management
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "current_module" not in st.session_state:
        st.session_state.current_module = None
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "module_initialized" not in st.session_state:
        st.session_state.module_initialized = {}


def set_current_module(module_id: str):
    """Set the current active module."""
    st.session_state.current_module = module_id
    if module_id not in st.session_state.messages:
        module_config = get_module_config(module_id)
        st.session_state.messages[module_id] = [
            {
                "role": "assistant",
                "content": f"Bonjour ! Je suis votre assistant spécialisé dans le **{module_config['name']}**. 🇲🇦\n\nPosez-moi vos questions et je vous répondrai en citant les articles pertinents."
            }
        ]


def go_back_to_home():
    """Return to the home page."""
    st.session_state.current_module = None


# =============================================================================
# Cached Resources
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_module_resources(module_id: str) -> Tuple[bool, Optional[str], int, Optional[VectorStoreManager]]:
    """
    Load and cache resources for a specific module.
    """
    try:
        module_config = get_module_config(module_id)
        
        # Create vector store manager (Qdrant)
        vs_manager = create_vector_store_manager(module_config)
        
        # Check if collection exists and has vectors
        collection_info = vs_manager.get_collection_info()
        
        if collection_info["exists"] and collection_info["count"] > 0:
            return True, None, collection_info["count"], vs_manager
        
        # Check if there are documents to process
        doc_processor = create_folder_processor(module_config)
        pdf_files = doc_processor.get_pdf_files()
        
        if not pdf_files:
            return False, (
                f"Aucun document trouvé dans le dossier '{module_config['documents_folder']}'. "
                f"Ajoutez des fichiers PDF et exécutez: python sync_documents.py --module {module_id}"
            ), 0, None
        
        return False, (
            f"La base vectorielle est vide. "
            f"Exécutez: python sync_documents.py --module {module_id}"
        ), 0, None
        
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "connect" in error_msg.lower():
            return False, (
                "Impossible de se connecter à Qdrant. "
                "Assurez-vous que Qdrant est en cours d'exécution."
            ), 0, None
        return False, f"Erreur inattendue: {error_msg}", 0, None


RAG_CHAIN_VERSION = "v7_qdrant"

@st.cache_resource(show_spinner=False)
def get_rag_chain(_vs_manager: VectorStoreManager, module_id: str, version: str = RAG_CHAIN_VERSION) -> RAGChainBuilder:
    """Get or create the RAG chain for a module."""
    module_config = get_module_config(module_id)
    return create_rag_chain(_vs_manager, module_config)


# =============================================================================
# Home Page
# =============================================================================

def render_home_page():
    """Render the home page matching the design provided."""
    
    # Optional Top Title (Hidden in screenshot but likely needed for branding)
    # st.markdown('<h1 class="main-title">IYYA</h1>', unsafe_allow_html=True)
    
    # 1. Header Section
    st.markdown('<div class="module-header">Choisissez votre module</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
    
    # 2. Module Grid (Using 4 columns to match the wide card look in the photo)
    # We create a centered layout with padding on sides if needed
    col_spacer_left, col_main, col_spacer_right = st.columns([0.5, 4, 0.5])
    
    with col_main:
        # Create 4 columns for the cards
        cols = st.columns(4, gap="medium")
        
        # Convert dictionary items to list for indexing
        modules_list = list(MODULES.items())
        
        for idx, (module_id, module_config) in enumerate(modules_list):
            # If we have more than 4 modules, wrap to next row (handled by modulo or just let streamlit flow)
            
            # Use modulo to place in columns
            col_idx = idx % 4
            
            with cols[col_idx]:
                is_enabled = module_config.get('enabled', True)
                
                # --- CARD SECTION ---
                # The visual card with Icon and Text
                with st.container(border=True):
                    # Icon
                    st.markdown(
                        f'<div class="card-icon">{module_config["icon"]}</div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Title
                    st.markdown(
                        f'<div class="card-title">{module_config["name"]}</div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Description
                    st.markdown(
                        f'<div class="card-desc">{module_config["description"]}</div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Badge for disabled
                    if not is_enabled:
                        st.markdown(
                            '<span style="background:#eee; padding:4px 8px; border-radius:4px; font-size:0.8rem; color:#666;">Bientôt disponible</span>', 
                            unsafe_allow_html=True
                        )

                # --- BUTTON SECTION ---
                # Button is OUTSIDE the card container to match the design (Gap between card and button)
                
                # Label: "Accéder au [ShortName]"
                btn_label = f"Accéder au {module_config['short_name']}"
                
                if st.button(
                    btn_label,
                    key=f"btn_{module_id}",
                    use_container_width=True,
                    disabled=not is_enabled
                ):
                    set_current_module(module_id)
                    st.rerun()
                
                # Spacer for vertical gap if multiple rows
                st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="powered-by">© 2025 IYYA - Assistant Juridique Marocain</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# Chat Page
# =============================================================================

def render_chat_page(module_id: str):
    """Render the chat interface for a specific module."""
    
    module_config = get_module_config(module_id)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### {module_config['icon']} {module_config['name']}")
        st.markdown("---")
        if st.button("← Changer de module", use_container_width=True):
            go_back_to_home()
            st.rerun()
    
    # Header
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Retour", key="back_btn"):
            go_back_to_home()
            st.rerun()
    
    with col_title:
        st.markdown(
            f'<h2 style="font-family: Playfair Display; color: #594A35; margin-top:0;">{module_config["icon"]} {module_config["name"]}</h2>',
            unsafe_allow_html=True
        )
    
    # Load module resources
    with st.spinner(f"Chargement du {module_config['short_name']}..."):
        success, error, num_chunks, vs_manager = load_module_resources(module_id)
    
    if not success:
        st.error(error)
        st.stop()
    
    # Initialize RAG
    rag_chain = get_rag_chain(vs_manager, module_id)
    query_handler = RAGQueryHandler(rag_chain, module_id)
    
    st.markdown("---")
    
    # Chat History
    for message in st.session_state.messages.get(module_id, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input
    if prompt := st.chat_input(f"Posez votre question..."):
        st.session_state.messages[module_id].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            history = st.session_state.messages.get(module_id, [])
            response = st.write_stream(query_handler.stream(prompt, conversation_history=history))
        
        st.session_state.messages[module_id].append({"role": "assistant", "content": response})


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    apply_golden_theme()
    init_session_state()
    
    if st.session_state.current_module is None:
        render_home_page()
    else:
        render_chat_page(st.session_state.current_module)


if __name__ == "__main__":
    main()