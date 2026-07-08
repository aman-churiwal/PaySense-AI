"""Tests for the PaySense AI agent."""

import chromadb
import pytest
from unittest.mock import patch
from src.config import Settings
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.sessions.session_manager import SessionManager
from src.agent.agent import PaySenseAgent


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel("all-MiniLM-L6-v2")


@pytest.fixture
def agent_setup(embedding_model):
    """Set up the full agent with seeded stores."""
    client = chromadb.EphemeralClient()
    kb_store = VectorStore(
        collection_name="test_kb_agent",
        persist_dir="",
        embedding_model=embedding_model,
        client=client
    )
    user_store = VectorStore(
        collection_name="test_user_agent",
        persist_dir="",
        embedding_model=embedding_model,
        client=client
    )

    # Seed KB
    kb_store.add_documents(
        "kb_hra",
        ["HRA stands for House Rent Allowance. It covers accommodation expenses."],
        [{"source": "knowledge_base", "topic": "hra"}],
    )

    retriever = Retriever(kb_store, user_store)
    session_mgr = SessionManager()
    settings = Settings(gemini_api_key="fake-key", chroma_persist_dir="")

    agent = PaySenseAgent(retriever, session_mgr, settings)
    session_id = session_mgr.create_session()

    yield agent, session_mgr, session_id


@patch("src.agent.agent.PaySenseAgent._call_llm")
def test_chat_returns_response(mock_llm, agent_setup):
    """Agent should return a string response."""
    mock_llm.return_value = "HRA stands for House Rent Allowance."
    agent, session_mgr, session_id = agent_setup
    response = agent.chat(session_id, "What is HRA?")
    assert isinstance(response, str)
    assert len(response) > 0


@patch("src.agent.agent.PaySenseAgent._call_llm")
def test_chat_stores_messages(mock_llm, agent_setup):
    """Agent should store both user and assistant messages."""
    mock_llm.return_value = "Your net pay is ₹42,258."
    agent, session_mgr, session_id = agent_setup
    agent.chat(session_id, "What is my net pay?")
    messages = session_mgr.get_messages(session_id)
    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles


@patch("src.agent.agent.PaySenseAgent._call_llm")
def test_chat_comparison_query(mock_llm, agent_setup):
    """Agent should detect comparison queries and use comparison logic."""
    mock_llm.return_value = "Your basic pay increased by 12%."
    agent, session_mgr, session_id = agent_setup

    # Add two documents to session
    session_mgr.add_document(session_id, "jan_payslip", "text", {
        "document_type": "payslip", "period": "Jan 2025",
        "basic_pay": 25000, "net_pay": 42000,
    })
    session_mgr.add_document(session_id, "jun_payslip", "text", {
        "document_type": "payslip", "period": "Jun 2025",
        "basic_pay": 28000, "net_pay": 47000,
    })

    response = agent.chat(session_id, "Compare my two payslips")
    assert isinstance(response, str)