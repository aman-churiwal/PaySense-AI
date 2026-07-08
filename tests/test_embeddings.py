"""Tests for the embedding model wrapper."""

import pytest
from src.rag.embeddings import EmbeddingModel


@pytest.fixture(scope="module")
def embedding_model():
    """Load the embedding model once for all tests in this module."""
    return EmbeddingModel("all-MiniLM-L6-v2")


def test_embed_returns_list_of_vectors(embedding_model):
    """Should return a list of float vectors."""
    result = embedding_model.embed(["hello world"])
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])


def test_embed_multiple_texts(embedding_model):
    """Should return one vector per input text."""
    texts = ["hello", "world", "test"]
    result = embedding_model.embed(texts)
    assert len(result) == 3


def test_embed_vector_dimension(embedding_model):
    """all-MiniLM-L6-v2 produces 384-dimensional vectors."""
    result = embedding_model.embed(["test"])
    assert len(result[0]) == 384


def test_embed_empty_list(embedding_model):
    """Should return empty list for empty input."""
    result = embedding_model.embed([])
    assert result == []