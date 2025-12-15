"""
FastAPI Backend for IYYA Flutter Web Application.
Provides REST API endpoints for the Flutter frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json

from config import MODULES, get_module_config
from document_loader import create_folder_processor
from vector_store import create_vector_store_manager, VectorStoreManager
from rag_chain import create_rag_chain, RAGChainBuilder, RAGQueryHandler

# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="IYYA API",
    description="Backend API for IYYA - Assistant Juridique Marocain",
    version="1.0.0"
)

# CORS middleware for Flutter Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# In-memory cache for loaded modules
# =============================================================================

_module_resources: Dict[str, Dict[str, Any]] = {}


def get_module_resources(module_id: str) -> Dict[str, Any]:
    """Load and cache module resources."""
    if module_id in _module_resources:
        return _module_resources[module_id]
    
    try:
        module_config = get_module_config(module_id)
        
        # Create vector store manager
        vs_manager = create_vector_store_manager(module_config)
        
        # Check collection info
        collection_info = vs_manager.get_collection_info()
        
        if not collection_info["exists"] or collection_info["count"] == 0:
            return {
                "success": False,
                "error": f"La base vectorielle est vide. Exécutez: python sync_documents.py --module {module_id}",
                "num_vectors": 0,
                "vs_manager": None,
                "rag_chain": None
            }
        
        # Create RAG chain
        rag_chain = create_rag_chain(vs_manager, module_config)
        query_handler = RAGQueryHandler(rag_chain, module_id)
        
        _module_resources[module_id] = {
            "success": True,
            "error": None,
            "num_vectors": collection_info["count"],
            "vs_manager": vs_manager,
            "rag_chain": rag_chain,
            "query_handler": query_handler
        }
        
        return _module_resources[module_id]
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "num_vectors": 0,
            "vs_manager": None,
            "rag_chain": None
        }


# =============================================================================
# Request/Response Models
# =============================================================================

class ModuleInfo(BaseModel):
    id: str
    name: str
    short_name: str
    description: str
    icon: str
    color: str
    enabled: bool


class ModulesResponse(BaseModel):
    modules: List[ModuleInfo]


class ModuleStatusResponse(BaseModel):
    success: bool
    error: Optional[str]
    num_vectors: int


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    module_id: str
    message: str
    conversation_history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str]
    error: Optional[str]


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "IYYA API is running"}


@app.get("/api/modules", response_model=ModulesResponse)
async def get_modules():
    """Get all available modules."""
    modules = []
    for module_id, config in MODULES.items():
        modules.append(ModuleInfo(
            id=module_id,
            name=config["name"],
            short_name=config["short_name"],
            description=config["description"],
            icon=config["icon"],
            color=config["color"],
            enabled=config.get("enabled", True)
        ))
    return ModulesResponse(modules=modules)


@app.get("/api/modules/{module_id}", response_model=ModuleInfo)
async def get_module(module_id: str):
    """Get a specific module's info."""
    if module_id not in MODULES:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
    
    config = MODULES[module_id]
    return ModuleInfo(
        id=module_id,
        name=config["name"],
        short_name=config["short_name"],
        description=config["description"],
        icon=config["icon"],
        color=config["color"],
        enabled=config.get("enabled", True)
    )


@app.get("/api/modules/{module_id}/status", response_model=ModuleStatusResponse)
async def get_module_status(module_id: str):
    """Get the status of a module (loaded, vector count, etc.)."""
    if module_id not in MODULES:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
    
    resources = get_module_resources(module_id)
    return ModuleStatusResponse(
        success=resources["success"],
        error=resources.get("error"),
        num_vectors=resources["num_vectors"]
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a chat message and get a response."""
    if request.module_id not in MODULES:
        raise HTTPException(status_code=404, detail=f"Module '{request.module_id}' not found")
    
    resources = get_module_resources(request.module_id)
    
    if not resources["success"]:
        return ChatResponse(
            success=False,
            answer=None,
            error=resources["error"]
        )
    
    try:
        query_handler: RAGQueryHandler = resources["query_handler"]
        
        # Convert conversation history
        history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
        
        # Get response
        result = query_handler.ask(request.message, conversation_history=history)
        
        return ChatResponse(
            success=result["success"],
            answer=result.get("answer"),
            error=result.get("error")
        )
        
    except Exception as e:
        return ChatResponse(
            success=False,
            answer=None,
            error=str(e)
        )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream a chat response."""
    if request.module_id not in MODULES:
        raise HTTPException(status_code=404, detail=f"Module '{request.module_id}' not found")
    
    resources = get_module_resources(request.module_id)
    
    if not resources["success"]:
        async def error_generator():
            yield f"data: {json.dumps({'error': resources['error']})}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    query_handler: RAGQueryHandler = resources["query_handler"]
    history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
    
    async def generate():
        try:
            for chunk in query_handler.stream(request.message, conversation_history=history):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# =============================================================================
# Run with: uvicorn api:app --reload --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
