"""Embedding model wrapper using sentence-transformers."""
from langgraph.store.base import embed
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """Wraps a sentence-transformers model for text embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model =  SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings into float vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """

        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]
