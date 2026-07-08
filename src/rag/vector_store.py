"""ChromaDB vector store for document storage and retrieval."""

import chromadb

from src.rag.embeddings import EmbeddingModel


class VectorStore:
    """Manages a ChromaDB collection for storing and querying document chunks."""

    def __init__(
            self,
            collection_name: str,
            persist_dir: str,
            embedding_model: EmbeddingModel,
            client=None,
    ):
        self._embedding_model = embedding_model
        self._client = client or chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
            self, doc_id: str, chunks: list[str], metadatas: list[dict]
    ) -> None:
        """
        Add document chunks to the vector store.

        Args:
            doc_id: Unique identifier for the document.
            chunks: List of text chunks to store.
            metadatas: List of metadata dict, one per chunk.
        """

        if not chunks:
            return

        embeddings = self._embedding_model.embed(chunks)

        # Tag each chunk with the doc_id for later deletion
        for meta in metadatas:
            meta["doc_id"] = doc_id

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Query the vector store for relevant chunks.

        Args:
            query_text: The search query.
            n_results: Maximum number of results to return.

        Returns:
            List of dict with keys: text, metadata, distance.
        """
        query_embedding = self._embedding_model.embed([query_text])

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self._collection.count()),
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        output = []
        for i in range(len(results["documents"][0])):
            output.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else {},
                }
            )
        return output

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document."""
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        """Returns the total number of chunks in the collection."""
        return self._collection.count()
