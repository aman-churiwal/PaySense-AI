"""Agent tools for searching knowledge base, user documents, and comparing documents."""

from src.rag.retriever import Retriever
from src.sessions.session_manager import SessionManager
from src.comparison.comparator import compare_documents


def search_knowledge(retriever: Retriever, query: str) -> str:
    """
    Search the payroll knowledge base for relevant information.

    Returns formatted context string from KB results.
    """
    results = retriever.retrieve(query, n_results=5)
    kb_results = [r for r in results if r["metadata"].get("source") == "knowledge_base"]

    if not kb_results:
        return "No relevant information found in the knowledge base."

    context_parts = []
    for r in kb_results:
        topic = r["metadata"].get("topic", "unknown")
        context_parts.append(f"[Source: {topic}]\n{r['text']}")

    return "\n\n---\n\n".join(context_parts)


def search_documents(retriever: Retriever, query: str) -> str:
    """
    Search the user's uploaded documents for relevant information.

    Returns formatted context string from user document results.
    """
    results = retriever.retrieve(query, n_results=5)
    user_results = [r for r in results if r["metadata"].get("source") == "user_upload"]

    if not user_results:
        return "No relevant information found in your uploaded documents."

    context_parts = []
    for r in user_results:
        doc_id = r["metadata"].get("doc_id", "unknown")
        context_parts.append(f"[Source: {doc_id}]\n{r['text']}")

    return "\n\n---\n\n".join(context_parts)


def compare_docs(session_manager: SessionManager, session_id: str, doc_id_a: str, doc_id_b: str
) -> dict:
    """
    Compare two uploaded documents from the session.

    Returns comparison result dict or error dict.
    """
    try:
        docs = session_manager.get_documents(session_id)
    except KeyError:
        return {"error": "Session not found."}

    if doc_id_a not in docs:
        return {"error": f"Document '{doc_id_a}' not found in your session."}
    if doc_id_b not in docs:
        return {"error": f"Document '{doc_id_b}' not found in your session."}

    fields_a = docs[doc_id_a]["fields"]
    fields_b = docs[doc_id_b]["fields"]

    return compare_documents(fields_a, fields_b)