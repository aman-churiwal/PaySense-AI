"""Tests for the compare API endpoint."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config import Settings


@pytest.fixture
def client():
    settings = Settings(gemini_api_key="test-key", chroma_persist_dir="./test_chroma_api_cmp")
    with patch("src.api.app.EmbeddingModel"):
        with patch("src.api.app.VectorStore") as mock_vs:
            mock_vs.return_value.count.return_value = 1
            with patch("src.api.app.load_knowledge_base"):
                with patch("src.agent.agent.PaySenseAgent._call_llm", return_value="Comparison explanation"):
                    app = create_app(settings)
                    yield TestClient(app)


def test_compare_missing_documents(client):
    """Should return 404 when documents don't exist."""
    session_resp = client.post("/api/session")
    session_id = session_resp.json()["session_id"]

    response = client.post("/api/compare", json={
        "session_id": session_id,
        "doc_id_a": "nonexistent_a",
        "doc_id_b": "nonexistent_b",
    })
    assert response.status_code == 404