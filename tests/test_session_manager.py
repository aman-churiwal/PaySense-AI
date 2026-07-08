"""Tests for in-memory session manager."""

import pytest
from src.sessions.session_manager import SessionManager


@pytest.fixture
def manager():
    return SessionManager()


def test_create_session_returns_uuid(manager):
    """Should return a valid UUID string."""
    sid = manager.create_session()
    assert isinstance(sid, str)
    assert len(sid) == 36  # UUID format: 8-4-4-4-12


def test_get_session_returns_dict(manager):
    """Should return session data dict."""
    sid = manager.create_session()
    data = manager.get_session(sid)
    assert isinstance(data, dict)
    assert "documents" in data
    assert "messages" in data


def test_get_session_invalid_id_raises(manager):
    """Should raise KeyError for unknown session."""
    with pytest.raises(KeyError):
        manager.get_session("nonexistent-id")


def test_add_and_get_document(manager):
    """Should store and retrieve documents by session."""
    sid = manager.create_session()
    manager.add_document(sid, "doc1", "raw text here", {"basic_pay": 25000})
    docs = manager.get_documents(sid)
    assert "doc1" in docs
    assert docs["doc1"]["raw_text"] == "raw text here"
    assert docs["doc1"]["fields"]["basic_pay"] == 25000


def test_add_and_get_messages(manager):
    """Should store and retrieve chat messages in order."""
    sid = manager.create_session()
    manager.add_message(sid, "user", "What is my salary?")
    manager.add_message(sid, "assistant", "Your net pay is ₹42,258.")
    messages = manager.get_messages(sid)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_delete_session(manager):
    """Should remove session completely."""
    sid = manager.create_session()
    manager.delete_session(sid)
    with pytest.raises(KeyError):
        manager.get_session(sid)


def test_sessions_are_isolated(manager):
    """Documents in one session should not appear in another."""
    sid1 = manager.create_session()
    sid2 = manager.create_session()
    manager.add_document(sid1, "doc1", "text1", {"basic_pay": 100})
    docs1 = manager.get_documents(sid1)
    docs2 = manager.get_documents(sid2)
    assert "doc1" in docs1
    assert "doc1" not in docs2