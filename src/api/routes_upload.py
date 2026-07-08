"""Upload endpoint — handles PDF and image file uploads."""

import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.api.models import UploadResponse
from src.parsers.pdf_parser import parse_pdf
from src.parsers.ocr_parser import parse_image
from src.parsers.field_extractor import extract_fields
from src.config import Settings

router = APIRouter()

# These will be injected at app startup
_session_manager = None
_user_vector_store = None
_settings = None


def init_upload_routes(session_manager, user_vector_store, settings: Settings):
    """Inject dependencies for the upload routes."""
    global _session_manager, _user_vector_store, _settings
    _session_manager = session_manager
    _user_vector_store = user_vector_store
    _settings = settings


@router.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Upload a payslip or offer letter (PDF/image) and extract fields."""
    # Validate session
    try:
        _session_manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate file size
    content = await file.read()
    if len(content) > _settings.upload_max_bytes:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    # Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _settings.supported_formats:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {ext}. Supported: {_settings.supported_formats}",
        )

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(content)
    tmp.close()

    try:
        # Parse document
        if ext == ".pdf":
            raw_text = parse_pdf(tmp.name)
        else:
            raw_text = parse_image(tmp.name)

        # Extract structured fields
        fields = extract_fields(raw_text, _settings.gemini_api_key, _settings.gemini_model_name)

        # Generate doc ID
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        # Store in session
        _session_manager.add_document(session_id, doc_id, raw_text, fields)

        # Store chunks in vector store for RAG
        chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 450)]
        metadatas = [{"source": "user_upload", "doc_id": doc_id, "session_id": session_id} for _ in chunks]
        _user_vector_store.add_documents(doc_id, chunks, metadatas)

        doc_type = fields.get("document_type", "document")
        return UploadResponse(
            doc_id=doc_id,
            fields=fields,
            message=f"Successfully processed {doc_type}. You can now ask questions about it.",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp.name)