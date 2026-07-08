"""Tests for the chat API endpoint."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config import Settings


@pytest.fixture
def client():
    settings = Settings(gemini_api_key="test-key", chroma_persist_dir="./test_chroma_api_chat")
    with patch("src.api.app.EmbeddingModel"):
        with patch("src.api.app.VectorStore") as mock_vs:
            mock_vs.return_value.count.return_value = 1
            with patch("src.api.app.load_knowledge_base"):
                with patch("src.agent.agent.PaySenseAgent._call_llm", return_value="Test response"):
                    app = create_app(settings)
                    yield TestClient(app)


def test_chat_endpoint(client):
    """POST /api/chat should return agent response."""
    # Create session first
    session_resp = client.post("/api/session")
    session_id = session_resp.json()["session_id"]

    # Send chat message
    response = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "What is HRA?"
    })
    assert response.status_code == 200
    assert "response" in response.json()


def test_chat_invalid_session(client):
    """Should return 404 for invalid session."""
    response = client.post("/api/chat", json={
        "session_id": "invalid-session-id",
        "message": "Hello"
    })
    assert response.status_code == 404