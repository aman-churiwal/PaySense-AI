"""Tests for application configuration."""

from src.config import Settings

def test_settings_defaults():
    """Settings should have sensible defaults."""
    s = Settings()
    assert s.chroma_persist_dir == "./chroma_data"
    assert s.embedding_model_name == "all-MiniLM-L6-v2"
    assert s.gemini_model_name == "gemini-2.0-flash"
    assert s.upload_max_bytes == 10_485_760
    assert ".pdf" in s.supported_formats
    assert ".png" in s.supported_formats

def test_settings_from_env(monkeypatch):
    """Settings.from_env() should read from environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "my-test-key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
    s = Settings.from_env()
    assert s.gemini_api_key == "my-test-key"
    assert s.chroma_persist_dir == "/tmp/test_chroma"

def test_settings_is_frozen():
    """Settings should be immutable."""
    s = Settings()
    try:
        s.gemini_api_key = "new-key"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass