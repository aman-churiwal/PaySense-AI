"""PaySense AI agent - orchestrates RAG retrieval, tools and LLM responses."""

import json
import google.generativeai as genai
from src.agent.prompts import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE, COMPARE_PROMPT_TEMPLATE
from src.agent.tools import search_knowledge, search_documents, compare_docs
from src.rag.retriever import Retriever
from src.sessions.session_manager import SessionManager
from src.config import Settings


class PaySenseAgent:
    """RAG-powered agent for payslip and offer letter analysis."""

    def __init__(
            self,
            retriever: Retriever,
            session_manager: SessionManager,
            settings: Settings,
    ):
        self._retriever = retriever
        self._session_manager = session_manager
        self._settings = settings


    def _call_llm(self, prompt: str) -> str:
        """Call Gemini LLM with the given prompt."""

        genai.configure(api_key=self._settings.gemini_api_key)
        model = genai.GenerativeModel(self._settings.gemini_model_name)
        response = model.generate_content(
            [{"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}]
        )
        return response.text


    def _format_document_fields(self, session_id: str) -> str:
        """Format all uploaded document fields for the prompt."""
        try:
            docs = self._session_manager.get_documents(session_id)

        except KeyError:
            return "No documents uploaded."

        if not docs:
            return "No documents uploaded."

        parts = []
        for doc_id, doc_data in docs.items():
            fields = doc_data.get("fields", {})
            parts.append(f"Document: {doc_id}\n{json.dumps(fields, indent=2, default=str)}")

        return "\n\n".join(parts)


    def _format_chat_history(self, session_id: str) -> str:
        """Format recent chat history for context."""
        try:
            messages = self._session_manager.get_messages(session_id)
        except KeyError:
            return ""

        # Keep last 10 messages to stay within context limits
        recent = messages[-10:]
        return "\n".join(f"{m['role']}: {m['content']}" for m in recent)


    def _is_comparison_query(self, message: str) -> bool:
        """Detect if the user is asking to compare documents."""
        comparison_keywords = ["compare", "difference", "differ", "change", "changed", "vs", "versus"]

        return any(kw in message.lower() for kw in comparison_keywords)


    def chat(self, session_id: str, user_message: str) -> str:
        """
        Process a user message and return the agent's response.

        Args:
            session_id: The user's session ID.
            user_message: The user's question or message.

        Returns:
            The agent's response string.
        """
        # Store user message

        self._session_manager.add_message(session_id, "user", user_message)

        # Check if this is a comparison query
        if self._is_comparison_query(user_message):
            docs = self._session_manager.get_documents(session_id)
            doc_ids = list(docs.keys())
            if len(doc_ids) >= 2:
                comparison = compare_docs(self._session_manager, session_id, doc_ids[0], doc_ids[1])
                if "error" not in comparison:
                    prompt = COMPARE_PROMPT_TEMPLATE.format(
                        comparison_data=json.dumps(comparison, indent=2, default=str),
                        doc_a_label=comparison["doc_a_label"],
                        doc_b_label=comparison["doc_b_label"],
                        question=user_message,
                    )
                    response = self._call_llm(prompt)
                    self._session_manager.add_message(session_id, "assistant", response)
                    return response

        # Standard RAG Q&A flow
        kb_context = search_knowledge(self._retriever, user_message)
        doc_context = search_documents(self._retriever, user_message)
        combined_context = f"Knowledge Base:\n{kb_context}\n\nUploaded Documents:\n{doc_context}"

        prompt = QA_PROMPT_TEMPLATE.format(
            context=combined_context,
            document_fields=self._format_document_fields(session_id),
            chat_history=self._format_chat_history(session_id),
            question=user_message,
        )

        response = self._call_llm(prompt)
        self._session_manager.add_message(session_id, "assistant", response)
        return response