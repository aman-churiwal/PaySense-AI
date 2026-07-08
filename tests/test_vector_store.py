"""Tests for ChromaDB vector store operations."""

import chromadb
import pytest
import uuid
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel("all-MiniLM-L6-v2")


@pytest.fixture
def vector_store(embedding_model):
    """Create a fresh in-memory vector store for each test."""
    client = chromadb.EphemeralClient()
    store = VectorStore(
        collection_name=f"test_{uuid.uuid4()}",
        persist_dir="",
        embedding_model=embedding_model,
        client=client,
    )
    yield store


def test_add_and_query_documents(vector_store):
    """Should add documents and retrieve relevant ones."""
    chunks = [
        "Basic Pay is the fixed component of salary.",
        "HRA is House Rent Allowance for accommodation expenses.",
        "PF stands for Provident Fund, a retirement savings scheme.",
    ]
    metadatas = [
        {"source": "kb", "topic": "basic_pay"},
        {"source": "kb", "topic": "hra"},
        {"source": "kb", "topic": "pf"},
    ]
    vector_store.add_documents("doc1", chunks, metadatas)
    results = vector_store.query("What is HRA?", n_results=2)
    assert len(results) <= 2
    assert any("HRA" in r["text"] for r in results)


def test_count_documents(vector_store):
    """Should return correct document count."""
    assert vector_store.count() == 0
    vector_store.add_documents("doc1", ["chunk1", "chunk2"], [{}, {}])
    assert vector_store.count() == 2


def test_delete_document(vector_store):
    """Should delete all chunks for a given doc_id."""
    vector_store.add_documents("doc_a", ["chunk1"], [{"doc_id": "doc_a"}])
    vector_store.add_documents("doc_b", ["chunk2"], [{"doc_id": "doc_b"}])
    assert vector_store.count() == 2
    vector_store.delete_document("doc_a")
    assert vector_store.count() == 1


def test_query_returns_expected_shape(vector_store):
    """Each result should have text, metadata, and distance."""
    vector_store.add_documents("doc1", ["test chunk"], [{"source": "test"}])
    results = vector_store.query("test", n_results=1)
    assert len(results) == 1
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "distance" in results[0]
