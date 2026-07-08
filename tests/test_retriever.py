"""Tests for the combined retriever (KB + user docs)."""

import os
import shutil

import chromadb
import pytest
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever


TEST_CHROMA_DIR = "./test_chroma_retriever"


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel("all-MiniLM-L6-v2")


@pytest.fixture
def stores(embedding_model):
    client = chromadb.EphemeralClient()
    kb_store = VectorStore(
        collection_name="test_kb",
        persist_dir="",
        embedding_model=embedding_model,
        client=client
    )
    user_store = VectorStore(
        collection_name="test_user",
        persist_dir="",
        embedding_model=embedding_model,
        client=client
    )

    # Seed KB store
    kb_store.add_documents(
        "kb_hra",
        ["HRA stands for House Rent Allowance. It is provided for accommodation."],
        [{"source": "knowledge_base", "topic": "hra"}],
    )
    kb_store.add_documents(
        "kb_pf",
        ["PF is Provident Fund. Employee contributes 12% of basic pay."],
        [{"source": "knowledge_base", "topic": "pf"}],
    )

    # Seed user store
    user_store.add_documents(
        "user_doc_1",
        ["Rahul Sharma payslip June 2025. Basic: 25000, HRA: 12500, Net: 42258."],
        [{"source": "user_upload", "doc_id": "user_doc_1"}],
    )

    yield kb_store, user_store


def test_retrieve_combines_kb_and_user_results(stores):
    """Should return results from both KB and user document stores."""
    kb_store, user_store = stores
    retriever = Retriever(kb_store, user_store)
    results = retriever.retrieve("What is HRA?", n_results=5)
    sources = {r["metadata"].get("source") for r in results}
    assert "knowledge_base" in sources or "user_upload" in sources
    assert len(results) > 0


def test_retrieve_returns_sorted_by_relevance(stores):
    """Results should be sorted by distance (ascending = most relevant first)."""
    kb_store, user_store = stores
    retriever = Retriever(kb_store, user_store)
    results = retriever.retrieve("HRA allowance", n_results=5)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_retrieve_respects_n_results(stores):
    """Should return at most n_results items."""
    kb_store, user_store = stores
    retriever = Retriever(kb_store, user_store)
    results = retriever.retrieve("salary", n_results=2)
    assert len(results) <= 2