"""Tests for the upload API endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config import Settings


@pytest.fixture
def client():
    settings = Settings(gemini_api_key="test-key", chroma_persist_dir="./test_chroma_api")
    with patch("src.api.app.EmbeddingModel"):
        with patch("src.api.app.VectorStore") as mock_vs:
            mock_vs.return_value.count.return_value = 1
            with patch("src.api.app.load_knowledge_base"):
                app = create_app(settings)
                yield TestClient(app)


def test_create_session(client):
    """POST /api/session should return a session_id."""
    response = client.post("/api/session")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 36