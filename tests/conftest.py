"""Shared test fixtures for PaySense AI."""

import pytest
from src.config import Settings

@pytest.fixture
def test_settings() -> Settings:
    """Settings configured for testing - no real API keys."""
    return Settings(
        gemini_api_key="test-key-not-real",
        chroma_persist_dir="./test_chroma_data",
        embedding_model_name="all-MiniLM-L6-v2",
        gemini_model_name="gemini-2.0-flash",
        upload_max_bytes=10_485_760,
    )