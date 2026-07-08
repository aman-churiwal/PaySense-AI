"""Chat endpoint - handles user questions via the RAG agent."""

from fastapi import APIRouter, HTTPException
from src.api.models import ChatRequest, ChatResponse

router = APIRouter()

_agent = None


def init_chat_routes(agent):
    """Inject the PaySense agent dependency."""
    global _agent
    _agent = agent


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the PaySense AI agent."""
    try:
        response = _agent.chat(request.session_id, request.message)
        return ChatResponse(response=response)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")