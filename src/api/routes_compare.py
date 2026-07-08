"""Compare endpoint - handles document comparison requests."""

import json
from fastapi import APIRouter, HTTPException
from src.api.models import CompareRequest, CompareResponse
from src.comparison.comparator import compare_documents

router = APIRouter()

_session_manager = None
_agent = None


def init_compare_routes(session_manager, agent):
    """Inject dependencies for comparison routes."""
    global _session_manager, _agent
    _session_manager = session_manager
    _agent = agent


@router.post("/api/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """Compare two uploaded documents."""
    try:
        docs = _session_manager.get_documents(request.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.doc_id_a not in docs:
        raise HTTPException(status_code=404, detail=f"Document '{request.doc_id_a}' not found")

    if request.doc_id_b not in docs:
        raise HTTPException(status_code=404, detail=f"Document '{request.doc_id_b}' not found")

    fields_a = docs[request.doc_id_a]["fields"]
    fields_b = docs[request.doc_id_b]["fields"]
    comparison = compare_documents(fields_a, fields_b)

    # Get LLM explanation via the agent
    explanation = _agent.chat(
        request.session_id,
        f"Compare documents: {json.dumps(comparison, default=str)}. User asked: {request.question}",
    )

    return CompareResponse(comparison = comparison, explanation=explanation)