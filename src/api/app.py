"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.models import SessionResponse
from src.api.routes_upload import router as upload_router, init_upload_routes
from src.api.routes_chat import router as chat_router, init_chat_routes
from src.api.routes_compare import router as compare_router, init_compare_routes
from src.config import Settings
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.knowledge_base import load_knowledge_base
from src.rag.retriever import Retriever
from src.sessions.session_manager import SessionManager
from src.agent.agent import PaySenseAgent


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings.from_env()

    app = FastAPI(
        title="PaySense AI",
        description="Smart Payslip & Offer Letter Assistant powered by RAG",
        version="1.0.0",
    )

    # Initialize components
    embedding_model = EmbeddingModel(settings.embedding_model_name)
    kb_store = VectorStore("payroll_kb", settings.chroma_persist_dir, embedding_model)
    user_store = VectorStore("user_documents", settings.chroma_persist_dir, embedding_model)

    # Load knowledge base (skip if already loaded)
    if kb_store.count() == 0:
        load_knowledge_base(kb_store, "./knowledge_base")

    retriever = Retriever(kb_store, user_store)
    session_manager = SessionManager()
    agent = PaySenseAgent(retriever, session_manager, settings)

    # Inject dependencies
    init_upload_routes(session_manager, user_store, settings)
    init_chat_routes(agent)
    init_compare_routes(session_manager, agent)

    # Register routers
    app.include_router(upload_router)
    app.include_router(chat_router)
    app.include_router(compare_router)

    # Session creation endpoint
    @app.post("/api/session", response_model=SessionResponse)
    async def create_session():
        session_id = session_manager.create_session()
        return SessionResponse(session_id=session_id)

    # Document list endpoint
    @app.get("/api/session/{session_id}/documents")
    async def get_documents(session_id: str):
        try:
            docs = session_manager.get_documents(session_id)
            # Return just fields (not raw text) for the frontend
            return {
                "documents": {
                    doc_id: {"fields": data["fields"]}
                    for doc_id, data in docs.items()
                }
            }
        except KeyError:
            return {"error": "Session not found"}

    # Serve frontend
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app