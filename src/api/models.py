"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    doc_id: str
    fields: dict
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


class CompareRequest(BaseModel):
    session_id: str
    doc_id_a: str
    doc_id_b: str
    question: str = "Compare these two documents"


class CompareResponse(BaseModel):
    comparison: dict
    explanation: str


class ErrorResponse(BaseModel):
    error: str