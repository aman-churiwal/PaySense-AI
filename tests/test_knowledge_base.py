"""Tests for payroll knowledge base loading."""

import os
import shutil

import chromadb
import pytest
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.knowledge_base import load_knowledge_base

KB_DIR = "./knowledge_base"


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel("all-MiniLM-L6-v2")


@pytest.fixture
def kb_vector_store(embedding_model):
    client = chromadb.EphemeralClient()
    store = VectorStore(
        collection_name="test_kb",
        persist_dir="",
        embedding_model=embedding_model,
        client=client,
    )
    yield store


def test_load_knowledge_base_indexes_documents(kb_vector_store):
    """Should load all KB Markdown files and index chunks."""
    count = load_knowledge_base(kb_vector_store, KB_DIR)
    assert count > 0
    assert kb_vector_store.count() > 0


def test_load_knowledge_base_returns_chunk_count(kb_vector_store):
    """Should return the number of chunks indexed."""
    count = load_knowledge_base(kb_vector_store, KB_DIR)
    assert isinstance(count, int)
    assert count == kb_vector_store.count()


def test_kb_query_finds_relevant_content(kb_vector_store):
    """After loading KB, queries should return relevant payroll info."""
    load_knowledge_base(kb_vector_store, KB_DIR)
    results = kb_vector_store.query("What is HRA and how is it calculated?", n_results=3)
    assert len(results) > 0
    # At least one result should mention HRA
    texts = " ".join(r["text"] for r in results)
    assert "HRA" in texts or "House Rent" in texts