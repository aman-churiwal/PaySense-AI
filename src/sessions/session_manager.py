"""In-memory session store for multi-user isolation."""

import uuid
from threading import Lock


class SessionManager:
    """Thread-safe in memory session manager."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._lock = Lock()


    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = {
                "documents": {},
                "messages": [],
            }
        return session_id


    def get_session(self, session_id: str) -> dict:
        """
        Get session data by ID.

        Raises:
            KeyError: If session does not exist.
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")

            return self._sessions[session_id]


    def add_document(self, session_id: str, doc_id: str, raw_text: str, fields: dict) -> None:
        """Add a parsed document to a session."""
        session = self.get_session(session_id)
        with self._lock:
            session["documents"][doc_id] = {
                "raw_text": raw_text,
                "fields": fields,
            }


    def get_documents(self, session_id: str) -> dict:
        """Get all documents in a session."""
        session = self.get_session(session_id)
        return session["documents"]


    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a chat message to a session's history."""
        session = self.get_session(session_id)
        with self._lock:
            session["messages"].append({"role": role, "content": content})


    def get_messages(self, session_id: str) -> list[dict]:
        """Get all chat messages in a session."""
        session = self.get_session(session_id)
        return list(session["messages"])


    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its data."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            else:
                raise KeyError(f"Session not found: {session_id}")