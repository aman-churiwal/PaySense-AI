"""Combined retriever that searches both knowledge base and user document stores."""

from src.rag.vector_store import VectorStore

class Retriever:
    """Retrieves relevant context from both KB and user document stores."""

    def __init__(self, kb_store: VectorStore, user_store: VectorStore):
        self._kb_store = kb_store
        self._user_store = user_store

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Searches both stores and return merged, deduplicated, sorted results.

        Args:
            query: The user's question or search query.
            n_results: Maximum total results to return.

        Returns:
            List of dicts sorted by distance (ascending = most relevant first).
            Each dict has keys: text, metadata, distnace.
        """
        kb_results = []
        user_results = []

        if self._kb_store.count() > 0:
            kb_results = self._kb_store.query(query, n_results=n_results)

        if self._user_store.count() > 0:
            user_results = self._user_store.query(query, n_results=n_results)

        # Merge and deduplicate by text content
        seen_texts = set()
        merged = []
        for result in kb_results + user_results:
            text = result['text']
            if text not in seen_texts:
                seen_texts.add(text)
                merged.append(result)

        # Sort by distance (lower = more relevant)
        merged.sort(key=lambda r: r["distance"])

        return merged[:n_results]