# PaySense AI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG-powered web app where users upload payslips/offer letters (PDF/image), ask questions in plain English, and compare two documents side-by-side — focused on Indian payroll.

**Architecture:** FastAPI backend with a LangChain agent that uses two ChromaDB collections (payroll knowledge base + user documents). Documents are parsed via PyMuPDF/Tesseract, fields extracted via Gemini Flash, and stored as both chunked text (for semantic search) and structured JSON (for calculations/comparisons). A clean HTML/CSS/JS frontend provides upload, chat, and comparison UIs.

**Tech Stack:**
- Python 3.11+, FastAPI, Uvicorn
- LangChain, Google Gemini 2.0 Flash (free tier)
- ChromaDB (local), sentence-transformers (`all-MiniLM-L6-v2`)
- PyMuPDF (`fitz`), Tesseract OCR (`pytesseract`)
- pytest for testing
- HTML/CSS/JS frontend (no framework)

## Global Constraints

- Python ≥ 3.11
- All LLM calls use Gemini 2.0 Flash free tier (15 RPM) — no paid APIs
- Embeddings use local `all-MiniLM-L6-v2` — zero API cost
- ChromaDB runs in-process (no server needed)
- No frontend framework — vanilla HTML/CSS/JS
- Every task ends with passing tests and a commit
- File size limit: 10 MB per upload
- Supported formats: PDF, PNG, JPG, JPEG

---

## File Structure

```
paysense-ai/
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── run.py                          # Entry point: uvicorn launcher
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Settings, env vars, constants
│   │
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py           # Extract text from PDFs (PyMuPDF)
│   │   ├── ocr_parser.py           # Extract text from images (Tesseract)
│   │   └── field_extractor.py      # LLM-based structured field extraction
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py           # Embedding model wrapper
│   │   ├── vector_store.py         # ChromaDB operations (add, query, delete)
│   │   ├── knowledge_base.py       # Load & index payroll KB docs
│   │   └── retriever.py            # Retrieval logic (KB + user docs)
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py                # Agent tools (search_docs, extract_fields, compare)
│   │   ├── prompts.py              # System prompts, prompt templates
│   │   └── agent.py                # LangChain agent setup & execution
│   │
│   ├── comparison/
│   │   ├── __init__.py
│   │   └── comparator.py           # Field-by-field diff + explanation logic
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI app factory
│   │   ├── routes_upload.py        # POST /upload endpoint
│   │   ├── routes_chat.py          # POST /chat endpoint
│   │   ├── routes_compare.py       # POST /compare endpoint
│   │   └── models.py               # Pydantic request/response schemas
│   │
│   └── sessions/
│       ├── __init__.py
│       └── session_manager.py      # In-memory session store
│
├── knowledge_base/
│   ├── ctc_breakdown.md
│   ├── basic_pay.md
│   ├── hra.md
│   ├── provident_fund.md
│   ├── esi.md
│   ├── tds_tax.md
│   ├── gratuity.md
│   ├── professional_tax.md
│   └── offer_letter_terms.md
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Shared fixtures
    ├── test_pdf_parser.py
    ├── test_ocr_parser.py
    ├── test_field_extractor.py
    ├── test_embeddings.py
    ├── test_vector_store.py
    ├── test_knowledge_base.py
    ├── test_retriever.py
    ├── test_comparator.py
    ├── test_session_manager.py
    ├── test_agent.py
    ├── test_api_upload.py
    ├── test_api_chat.py
    └── test_api_compare.py
```

---

## Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `paysense-ai/requirements.txt`
- Create: `paysense-ai/.env.example`
- Create: `paysense-ai/.gitignore`
- Create: `paysense-ai/src/__init__.py`
- Create: `paysense-ai/src/config.py`
- Create: `paysense-ai/tests/__init__.py`
- Create: `paysense-ai/tests/conftest.py`
- Test: `paysense-ai/tests/conftest.py` (smoke test via import)

**Interfaces:**
- Consumes: Nothing (first task)
- Produces: `src.config.Settings` dataclass with fields: `gemini_api_key: str`, `chroma_persist_dir: str`, `embedding_model_name: str`, `upload_max_bytes: int`, `supported_formats: list[str]`, `gemini_model_name: str`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.venv/
venv/
chroma_data/
uploads/
.pytest_cache/
```

- [ ] **Step 2: Create `requirements.txt`**

```
fastapi==0.115.0
uvicorn==0.30.0
python-dotenv==1.0.1
langchain==0.3.0
langchain-google-genai==2.0.0
chromadb==0.5.0
sentence-transformers==3.0.0
PyMuPDF==1.24.0
pytesseract==0.3.10
Pillow==10.4.0
pydantic==2.9.0
pytest==8.3.0
httpx==0.27.0
```

- [ ] **Step 3: Create `.env.example`**

```
GEMINI_API_KEY=your-gemini-api-key-here
CHROMA_PERSIST_DIR=./chroma_data
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
GEMINI_MODEL_NAME=gemini-2.0-flash
UPLOAD_MAX_BYTES=10485760
```

- [ ] **Step 4: Create `src/__init__.py`**

```python
# PaySense AI - Smart Payslip & Offer Letter Assistant
```

- [ ] **Step 5: Create `src/config.py`**

```python
"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    gemini_api_key: str = ""
    chroma_persist_dir: str = "./chroma_data"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    gemini_model_name: str = "gemini-2.0-flash"
    upload_max_bytes: int = 10_485_760  # 10 MB
    supported_formats: list[str] = field(
        default_factory=lambda: [".pdf", ".png", ".jpg", ".jpeg"]
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Create Settings from environment variables."""
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
            upload_max_bytes=int(os.getenv("UPLOAD_MAX_BYTES", "10485760")),
        )
```

- [ ] **Step 6: Create `tests/__init__.py`**

```python
# PaySense AI Tests
```

- [ ] **Step 7: Create `tests/conftest.py`**

```python
"""Shared test fixtures for PaySense AI."""

import pytest
from src.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Settings configured for testing — no real API keys."""
    return Settings(
        gemini_api_key="test-key-not-real",
        chroma_persist_dir="./test_chroma_data",
        embedding_model_name="all-MiniLM-L6-v2",
        gemini_model_name="gemini-2.0-flash",
        upload_max_bytes=10_485_760,
    )
```

- [ ] **Step 8: Write smoke test for config**

Create `tests/test_config.py`:

```python
"""Tests for application configuration."""

from src.config import Settings


def test_settings_defaults():
    """Settings should have sensible defaults."""
    s = Settings()
    assert s.chroma_persist_dir == "./chroma_data"
    assert s.embedding_model_name == "all-MiniLM-L6-v2"
    assert s.gemini_model_name == "gemini-2.0-flash"
    assert s.upload_max_bytes == 10_485_760
    assert ".pdf" in s.supported_formats
    assert ".png" in s.supported_formats


def test_settings_from_env(monkeypatch):
    """Settings.from_env() should read from environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "my-test-key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
    s = Settings.from_env()
    assert s.gemini_api_key == "my-test-key"
    assert s.chroma_persist_dir == "/tmp/test_chroma"


def test_settings_is_frozen():
    """Settings should be immutable."""
    s = Settings()
    try:
        s.gemini_api_key = "new-key"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass
```

- [ ] **Step 9: Run tests**

Run: `cd paysense-ai && python -m pytest tests/test_config.py -v`
Expected: 3 tests PASS

- [ ] **Step 10: Commit**

```bash
git init
git add .
git commit -m "feat: project scaffolding with config and test setup"
```

---

## Task 2: PDF Parser

**Files:**
- Create: `paysense-ai/src/parsers/__init__.py`
- Create: `paysense-ai/src/parsers/pdf_parser.py`
- Test: `paysense-ai/tests/test_pdf_parser.py`
- Create: `paysense-ai/tests/fixtures/sample_payslip.pdf` (generated in test setup)

**Interfaces:**
- Consumes: Nothing (standalone utility)
- Produces: `parse_pdf(file_path: str) -> str` — returns extracted text from a PDF file. Raises `ValueError` if file is not a PDF or unreadable.

- [ ] **Step 1: Create `src/parsers/__init__.py`**

```python
"""Document parsers for PDF and image files."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_pdf_parser.py`:

```python
"""Tests for PDF text extraction."""

import os
import tempfile
import fitz  # PyMuPDF — used to create test PDFs
import pytest
from src.parsers.pdf_parser import parse_pdf


@pytest.fixture
def sample_pdf_path():
    """Create a minimal PDF with known text content."""
    doc = fitz.open()
    page = doc.new_page()
    text = "Employee Name: Rahul Sharma\nBasic Pay: 25000\nHRA: 12500\nPF: 3000\nNet Pay: 34500"
    page.insert_text((72, 72), text, fontsize=12)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    doc.close()
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


def test_parse_pdf_extracts_text(sample_pdf_path):
    """Should extract readable text from a valid PDF."""
    result = parse_pdf(sample_pdf_path)
    assert "Rahul Sharma" in result
    assert "Basic Pay" in result
    assert "25000" in result


def test_parse_pdf_returns_string(sample_pdf_path):
    """Should return a string."""
    result = parse_pdf(sample_pdf_path)
    assert isinstance(result, str)
    assert len(result) > 0


def test_parse_pdf_invalid_file():
    """Should raise ValueError for non-PDF files."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("not a pdf")
        f.flush()
        try:
            with pytest.raises(ValueError, match="Could not parse"):
                parse_pdf(f.name)
        finally:
            os.unlink(f.name)


def test_parse_pdf_missing_file():
    """Should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        parse_pdf("/nonexistent/path/file.pdf")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_pdf_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.parsers.pdf_parser'`

- [ ] **Step 4: Write minimal implementation**

Create `src/parsers/pdf_parser.py`:

```python
"""Extract text content from PDF files using PyMuPDF."""

import os
import fitz  # PyMuPDF


def parse_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Extracted text as a single string, pages separated by newlines.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as a PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Could not parse file as PDF: {file_path}. Error: {e}")

    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"Could not parse file as PDF: {file_path}. No pages found.")

    pages_text = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text.strip())
    doc.close()

    result = "\n\n".join(pages_text)
    if not result.strip():
        raise ValueError(
            f"Could not parse file as PDF: {file_path}. No extractable text found."
        )

    return result
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_pdf_parser.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/parsers/ tests/test_pdf_parser.py
git commit -m "feat: add PDF text extraction with PyMuPDF"
```

---

## Task 3: OCR Parser

**Files:**
- Create: `paysense-ai/src/parsers/ocr_parser.py`
- Test: `paysense-ai/tests/test_ocr_parser.py`

**Interfaces:**
- Consumes: Nothing (standalone utility)
- Produces: `parse_image(file_path: str) -> str` — returns OCR-extracted text from an image file. Raises `ValueError` if file is not a supported image or OCR fails.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ocr_parser.py`:

```python
"""Tests for OCR image text extraction."""

import os
import tempfile
import pytest
from PIL import Image, ImageDraw, ImageFont
from src.parsers.ocr_parser import parse_image


@pytest.fixture
def sample_image_path():
    """Create a minimal image with known text content."""
    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    # Use default font — no external font files needed
    draw.text((10, 10), "Employee Name: Rahul Sharma", fill="black")
    draw.text((10, 40), "Basic Pay: 25000", fill="black")
    draw.text((10, 70), "Net Pay: 34500", fill="black")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


def test_parse_image_extracts_text(sample_image_path):
    """Should extract readable text from a valid image."""
    result = parse_image(sample_image_path)
    # OCR may not be pixel-perfect, but key terms should appear
    assert isinstance(result, str)
    assert len(result) > 0


def test_parse_image_returns_string(sample_image_path):
    """Should return a non-empty string."""
    result = parse_image(sample_image_path)
    assert isinstance(result, str)


def test_parse_image_missing_file():
    """Should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        parse_image("/nonexistent/path/image.png")


def test_parse_image_invalid_file():
    """Should raise ValueError for non-image files."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w") as f:
        f.write("not an image")
        f.flush()
        try:
            with pytest.raises(ValueError, match="Could not parse"):
                parse_image(f.name)
        finally:
            os.unlink(f.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ocr_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.parsers.ocr_parser'`

- [ ] **Step 3: Write minimal implementation**

Create `src/parsers/ocr_parser.py`:

```python
"""Extract text from images using Tesseract OCR."""

import os
import pytesseract
from PIL import Image


def parse_image(file_path: str) -> str:
    """
    Extract text from an image file using Tesseract OCR.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        Extracted text as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as an image.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        img = Image.open(file_path)
    except Exception as e:
        raise ValueError(f"Could not parse file as image: {file_path}. Error: {e}")

    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        raise ValueError(f"Could not parse file as image: {file_path}. OCR error: {e}")

    if not text.strip():
        raise ValueError(
            f"Could not parse file as image: {file_path}. No text detected."
        )

    return text.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ocr_parser.py -v`
Expected: 4 tests PASS

> **Note:** Tesseract must be installed on the system. On Windows: download from https://github.com/tesseract-ocr/tesseract. On Ubuntu: `sudo apt install tesseract-ocr`. OCR accuracy with default fonts in test images may vary — the tests check for non-empty output rather than exact text matches.

- [ ] **Step 5: Commit**

```bash
git add src/parsers/ocr_parser.py tests/test_ocr_parser.py
git commit -m "feat: add OCR image text extraction with Tesseract"
```

---

## Task 4: LLM-Based Field Extractor

**Files:**
- Create: `paysense-ai/src/parsers/field_extractor.py`
- Test: `paysense-ai/tests/test_field_extractor.py`

**Interfaces:**
- Consumes: Raw text string (output of `parse_pdf` or `parse_image`)
- Produces: `extract_fields(raw_text: str, api_key: str, model_name: str) -> dict` — returns a dictionary of structured payslip/offer letter fields. Keys: `document_type` (`"payslip"` | `"offer_letter"` | `"unknown"`), `employee_name`, `basic_pay`, `hra`, `da`, `pf_employee`, `pf_employer`, `esi`, `tds`, `professional_tax`, `gross_salary`, `net_pay`, `ctc`, `other_allowances`, `other_deductions`, `period` (month/year), `company_name`. Missing fields are `None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_extractor.py`:

```python
"""Tests for LLM-based field extraction from document text."""

import json
import pytest
from unittest.mock import patch, MagicMock
from src.parsers.field_extractor import extract_fields, EXTRACTION_PROMPT


SAMPLE_PAYSLIP_TEXT = """
PAYSLIP - June 2025
Company: TechCorp India Pvt Ltd
Employee: Rahul Sharma
Employee ID: EMP-1234

Earnings:
  Basic Pay:         25,000
  HRA:               12,500
  DA:                 5,000
  Special Allowance:  7,500
  Gross Salary:      50,000

Deductions:
  PF (Employee):      3,000
  ESI:                  375
  Professional Tax:     200
  TDS:                4,167
  Total Deductions:   7,742

Net Pay: 42,258
"""

MOCK_LLM_RESPONSE = json.dumps({
    "document_type": "payslip",
    "employee_name": "Rahul Sharma",
    "company_name": "TechCorp India Pvt Ltd",
    "period": "June 2025",
    "basic_pay.md": 25000,
    "hra": 12500,
    "da": 5000,
    "pf_employee": 3000,
    "pf_employer": None,
    "esi": 375,
    "tds": 4167,
    "professional_tax": 200,
    "gross_salary": 50000,
    "net_pay": 42258,
    "ctc": None,
    "other_allowances": {"special_allowance": 7500},
    "other_deductions": {},
})


@patch("src.parsers.field_extractor._call_gemini")
def test_extract_fields_payslip(mock_call):
    """Should extract structured fields from payslip text."""
    mock_call.return_value = MOCK_LLM_RESPONSE
    result = extract_fields(SAMPLE_PAYSLIP_TEXT, "fake-key", "gemini-2.0-flash")
    assert result["document_type"] == "payslip"
    assert result["employee_name"] == "Rahul Sharma"
    assert result["basic_pay.md"] == 25000
    assert result["net_pay"] == 42258
    assert result["hra"] == 12500


@patch("src.parsers.field_extractor._call_gemini")
def test_extract_fields_returns_dict(mock_call):
    """Should always return a dictionary."""
    mock_call.return_value = MOCK_LLM_RESPONSE
    result = extract_fields(SAMPLE_PAYSLIP_TEXT, "fake-key", "gemini-2.0-flash")
    assert isinstance(result, dict)
    assert "document_type" in result


@patch("src.parsers.field_extractor._call_gemini")
def test_extract_fields_handles_malformed_llm_response(mock_call):
    """Should return a fallback dict when LLM returns non-JSON."""
    mock_call.return_value = "I can't parse this document properly."
    result = extract_fields(SAMPLE_PAYSLIP_TEXT, "fake-key", "gemini-2.0-flash")
    assert result["document_type"] == "unknown"


def test_extract_fields_empty_text():
    """Should raise ValueError for empty text."""
    with pytest.raises(ValueError, match="empty"):
        extract_fields("", "fake-key", "gemini-2.0-flash")


def test_extraction_prompt_contains_field_names():
    """The prompt should instruct the LLM to extract all required fields."""
    assert "basic_pay.md" in EXTRACTION_PROMPT
    assert "hra" in EXTRACTION_PROMPT
    assert "net_pay" in EXTRACTION_PROMPT
    assert "document_type" in EXTRACTION_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_field_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.parsers.field_extractor'`

- [ ] **Step 3: Write minimal implementation**

Create `src/parsers/field_extractor.py`:

```python
"""Extract structured payslip/offer letter fields using Gemini LLM."""

import json
import google.generativeai as genai


EXTRACTION_PROMPT = """You are a payroll document parser. Extract structured fields from the following document text.

Return ONLY a valid JSON object with these exact keys:
{
  "document_type": "payslip" or "offer_letter" or "unknown",
  "employee_name": string or null,
  "company_name": string or null,
  "period": string or null (e.g. "June 2025"),
  "basic_pay.md": number or null,
  "hra": number or null,
  "da": number or null,
  "pf_employee": number or null,
  "pf_employer": number or null,
  "esi": number or null,
  "tds": number or null,
  "professional_tax": number or null,
  "gross_salary": number or null,
  "net_pay": number or null,
  "ctc": number or null,
  "other_allowances": object with name:amount pairs or {},
  "other_deductions": object with name:amount pairs or {}
}

Rules:
- All monetary values should be numbers (no commas, no currency symbols)
- If a field is not found in the document, use null
- Do NOT invent or calculate values — only extract what is explicitly stated
- Return ONLY the JSON object, no explanation or markdown

Document text:
"""

_FALLBACK_FIELDS = {
    "document_type": "unknown",
    "employee_name": None,
    "company_name": None,
    "period": None,
    "basic_pay.md": None,
    "hra": None,
    "da": None,
    "pf_employee": None,
    "pf_employer": None,
    "esi": None,
    "tds": None,
    "professional_tax": None,
    "gross_salary": None,
    "net_pay": None,
    "ctc": None,
    "other_allowances": {},
    "other_deductions": {},
}


def _call_gemini(prompt: str, api_key: str, model_name: str) -> str:
    """Call Google Gemini API and return the response text."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


def extract_fields(raw_text: str, api_key: str, model_name: str) -> dict:
    """
    Extract structured fields from raw payslip/offer letter text using Gemini.

    Args:
        raw_text: The full text content of the document.
        api_key: Google Gemini API key.
        model_name: Gemini model name (e.g. "gemini-2.0-flash").

    Returns:
        Dictionary of extracted fields.

    Raises:
        ValueError: If raw_text is empty.
    """
    if not raw_text.strip():
        raise ValueError("Cannot extract fields from empty text.")

    prompt = EXTRACTION_PROMPT + raw_text
    response_text = _call_gemini(prompt, api_key, model_name)

    # Strip markdown code fences if present
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        fields = json.loads(cleaned)
        # Ensure all expected keys exist
        for key in _FALLBACK_FIELDS:
            if key not in fields:
                fields[key] = _FALLBACK_FIELDS[key]
        return fields
    except (json.JSONDecodeError, TypeError):
        return dict(_FALLBACK_FIELDS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_field_extractor.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/parsers/field_extractor.py tests/test_field_extractor.py
git commit -m "feat: add LLM-based structured field extraction for payslips"
```

---

## Task 5: Embeddings & Vector Store

**Files:**
- Create: `paysense-ai/src/rag/__init__.py`
- Create: `paysense-ai/src/rag/embeddings.py`
- Create: `paysense-ai/src/rag/vector_store.py`
- Test: `paysense-ai/tests/test_embeddings.py`
- Test: `paysense-ai/tests/test_vector_store.py`

**Interfaces:**
- Consumes: `src.config.Settings` (for `embedding_model_name`, `chroma_persist_dir`)
- Produces:
  - `EmbeddingModel(model_name: str)` with method `embed(texts: list[str]) -> list[list[float]]`
  - `VectorStore(collection_name: str, persist_dir: str, embedding_model: EmbeddingModel)` with methods:
    - `add_documents(doc_id: str, chunks: list[str], metadatas: list[dict]) -> None`
    - `query(query_text: str, n_results: int = 5) -> list[dict]` — returns list of `{"text": str, "metadata": dict, "distance": float}`
    - `delete_document(doc_id: str) -> None`
    - `count() -> int`

- [ ] **Step 1: Create `src/rag/__init__.py`**

```python
"""RAG pipeline — embeddings, vector store, knowledge base, retrieval."""
```

- [ ] **Step 2: Write the failing tests for embeddings**

Create `tests/test_embeddings.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write embeddings implementation**

Create `src/rag/embeddings.py`:

```python
"""Embedding model wrapper using sentence-transformers."""

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wraps a sentence-transformers model for text embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

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
```

- [ ] **Step 5: Run embedding tests**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Write the failing tests for vector store**

Create `tests/test_vector_store.py`:

```python
"""Tests for ChromaDB vector store operations."""

import os
import shutil
import pytest
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


TEST_CHROMA_DIR = "./test_chroma_vectorstore"


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel("all-MiniLM-L6-v2")


@pytest.fixture
def vector_store(embedding_model):
    """Create a fresh vector store for each test."""
    store = VectorStore(
        collection_name="test_collection",
        persist_dir=TEST_CHROMA_DIR,
        embedding_model=embedding_model,
    )
    yield store
    # Cleanup
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR)


def test_add_and_query_documents(vector_store):
    """Should add documents and retrieve relevant ones."""
    chunks = [
        "Basic Pay is the fixed component of salary.",
        "HRA is House Rent Allowance for accommodation expenses.",
        "PF stands for Provident Fund, a retirement savings scheme.",
    ]
    metadatas = [
        {"source": "kb", "topic": "basic_pay.md"},
        {"source": "kb", "topic": "hra"},
        {"source": "kb", "topic": "pf"},
    ]
    vector_store.add_documents("doc1", chunks, metadatas)
    results = vector_store.query("What is HRA?", n_results=2)
    assert len(results) <= 2
    assert any("HRA" in r["text"] for r in results)


def test_count_documents(vector_store):
    """Should return correct document count."""
    assert vector_store.count() == 0
    vector_store.add_documents("doc1", ["chunk1", "chunk2"], [{}, {}])
    assert vector_store.count() == 2


def test_delete_document(vector_store):
    """Should delete all chunks for a given doc_id."""
    vector_store.add_documents("doc_a", ["chunk1"], [{"doc_id": "doc_a"}])
    vector_store.add_documents("doc_b", ["chunk2"], [{"doc_id": "doc_b"}])
    assert vector_store.count() == 2
    vector_store.delete_document("doc_a")
    assert vector_store.count() == 1


def test_query_returns_expected_shape(vector_store):
    """Each result should have text, metadata, and distance."""
    vector_store.add_documents("doc1", ["test chunk"], [{"source": "test"}])
    results = vector_store.query("test", n_results=1)
    assert len(results) == 1
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "distance" in results[0]
```

- [ ] **Step 7: Run test to verify it fails**

Run: `python -m pytest tests/test_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 8: Write vector store implementation**

Create `src/rag/vector_store.py`:

```python
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
    ):
        self._embedding_model = embedding_model
        self._client = chromadb.PersistentClient(path=persist_dir)
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
            metadatas: List of metadata dicts, one per chunk.
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
            List of dicts with keys: text, metadata, distance.
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
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                }
            )
        return output

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document."""
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        return self._collection.count()
```

- [ ] **Step 9: Run vector store tests**

Run: `python -m pytest tests/test_vector_store.py -v`
Expected: 4 tests PASS

- [ ] **Step 10: Commit**

```bash
git add src/rag/ tests/test_embeddings.py tests/test_vector_store.py
git commit -m "feat: add embedding model and ChromaDB vector store"
```

---

## Task 6: Payroll Knowledge Base

**Files:**
- Create: `paysense-ai/knowledge_base/ctc_breakdown.md`
- Create: `paysense-ai/knowledge_base/basic_pay.md`
- Create: `paysense-ai/knowledge_base/hra.md`
- Create: `paysense-ai/knowledge_base/provident_fund.md`
- Create: `paysense-ai/knowledge_base/esi.md`
- Create: `paysense-ai/knowledge_base/tds_tax.md`
- Create: `paysense-ai/knowledge_base/gratuity.md`
- Create: `paysense-ai/knowledge_base/professional_tax.md`
- Create: `paysense-ai/knowledge_base/offer_letter_terms.md`
- Create: `paysense-ai/src/rag/knowledge_base.py`
- Test: `paysense-ai/tests/test_knowledge_base.py`

**Interfaces:**
- Consumes: `VectorStore` from Task 5
- Produces: `load_knowledge_base(vector_store: VectorStore, kb_dir: str) -> int` — loads all markdown files from the KB directory into the vector store, returns the number of chunks indexed.

- [ ] **Step 1: Create knowledge base markdown files**

Create `knowledge_base/ctc_breakdown.md`:

```markdown
# CTC (Cost to Company) Breakdown

CTC stands for Cost to Company. It is the total amount a company spends on an employee per year. CTC is NOT the same as in-hand salary (take-home pay).

## CTC Components
CTC typically includes:
- **Direct Benefits**: Basic Pay, HRA, Dearness Allowance (DA), Special Allowance, Conveyance Allowance, Medical Allowance
- **Indirect Benefits**: Employer PF contribution, Employer ESI contribution, Gratuity
- **Variable Components**: Performance bonus, incentives

## CTC vs In-Hand Salary
In-hand salary (net pay) = Gross Salary - Deductions
Where:
- Gross Salary = CTC - Employer PF - Employer ESI - Gratuity provision
- Deductions = Employee PF + ESI + Professional Tax + TDS (income tax)

## Example
If CTC = ₹10,00,000/year:
- Employer PF (12% of Basic): ~₹1,08,000
- Gratuity (4.81% of Basic): ~₹43,300
- Gross Salary: ~₹8,48,700
- After employee deductions: In-hand ~₹65,000-70,000/month
```

Create `knowledge_base/basic_pay.md`:

```markdown
# Basic Pay

Basic Pay is the core fixed component of an employee's salary. It is the foundation on which many other components are calculated.

## Importance
- PF (Provident Fund) is calculated as 12% of Basic Pay
- Gratuity is calculated based on Basic Pay
- HRA exemption depends on Basic Pay
- Basic Pay is fully taxable

## Typical Range
- Basic Pay is usually 40-50% of CTC
- Some companies keep it lower (30-35%) to reduce PF liability
- Government organizations often have higher Basic Pay percentages

## Impact on Take-Home
Higher Basic Pay means:
- More PF contribution (both employee and employer) — good for retirement, reduces take-home
- Higher gratuity eligibility
- Potentially higher HRA exemption
```

Create `knowledge_base/hra.md`:

```markdown
# HRA (House Rent Allowance)

HRA is an allowance paid by employers to employees to cover rental accommodation expenses.

## Tax Exemption (Section 10(13A))
HRA exemption is the MINIMUM of:
1. Actual HRA received
2. Rent paid minus 10% of Basic Pay
3. 50% of Basic Pay (metro cities: Delhi, Mumbai, Kolkata, Chennai) OR 40% of Basic Pay (non-metro cities)

## Example Calculation
- Basic Pay: ₹25,000/month
- HRA received: ₹12,500/month
- Rent paid: ₹15,000/month
- City: Bangalore (non-metro)

Exemption = min(₹12,500, ₹15,000 - ₹2,500, ₹10,000) = min(₹12,500, ₹12,500, ₹10,000) = ₹10,000
Taxable HRA = ₹12,500 - ₹10,000 = ₹2,500

## Key Points
- HRA is part of Gross Salary
- If you don't pay rent, entire HRA is taxable
- You need rent receipts to claim HRA exemption
```

Create `knowledge_base/provident_fund.md`:

```markdown
# Provident Fund (PF / EPF)

EPF (Employee Provident Fund) is a retirement savings scheme managed by EPFO.

## Contribution Rates
- **Employee contribution**: 12% of Basic Pay (mandatory)
- **Employer contribution**: 12% of Basic Pay, split as:
  - 3.67% to EPF account
  - 8.33% to EPS (Employee Pension Scheme), capped at ₹15,000 Basic

## Eligibility
- Mandatory for organizations with 20+ employees
- Applicable when Basic Pay ≤ ₹15,000/month (but most companies extend it to all)

## VPF (Voluntary Provident Fund)
- Employees can contribute more than 12% voluntarily
- Same interest rate as EPF (~8.15% per annum)
- Good for tax-free savings under Section 80C

## Key Points
- PF reduces take-home but builds retirement corpus
- Employee PF contribution qualifies for 80C deduction (up to ₹1.5 lakh)
- PF interest is tax-free up to ₹2.5 lakh/year contribution
- PF withdrawal before 5 years of service is taxable
```

Create `knowledge_base/esi.md`:

```markdown
# ESI (Employee State Insurance)

ESI is a social security scheme providing medical and cash benefits to employees.

## Eligibility
- Applicable when gross salary ≤ ₹21,000/month
- Once covered, remains covered for the contribution period even if salary exceeds ₹21,000

## Contribution Rates
- **Employee**: 0.75% of gross salary
- **Employer**: 3.25% of gross salary

## Benefits
- Medical treatment for employee and family
- Maternity benefit
- Sickness benefit
- Disablement benefit

## Key Points
- ESI is NOT applicable for employees earning > ₹21,000/month gross
- If ESI doesn't apply, the company may offer group health insurance instead
- ESI contribution is deducted from gross salary
```

Create `knowledge_base/tds_tax.md`:

```markdown
# TDS (Tax Deducted at Source) / Income Tax

TDS is the income tax deducted by the employer from the employee's salary each month.

## How Monthly TDS is Calculated
1. Estimate annual taxable income (gross salary - exemptions - deductions)
2. Calculate tax using applicable tax slab rates
3. Divide annual tax by 12 for monthly TDS

## Tax Regimes (FY 2025-26)
**New Regime (Default):**
- Up to ₹3,00,000: Nil
- ₹3,00,001 - ₹7,00,000: 5%
- ₹7,00,001 - ₹10,00,000: 10%
- ₹10,00,001 - ₹12,00,000: 15%
- ₹12,00,001 - ₹15,00,000: 20%
- Above ₹15,00,000: 30%
- Standard deduction: ₹75,000

**Old Regime:**
- Up to ₹2,50,000: Nil
- ₹2,50,001 - ₹5,00,000: 5%
- ₹5,00,001 - ₹10,00,000: 20%
- Above ₹10,00,000: 30%
- Allows deductions: 80C, 80D, HRA exemption, etc.

## Key Deductions (Old Regime)
- **Section 80C** (up to ₹1.5 lakh): PF, PPF, ELSS, life insurance, tuition fees
- **Section 80D** (up to ₹25,000): Health insurance premium
- **HRA exemption**: Under Section 10(13A)
```

Create `knowledge_base/gratuity.md`:

```markdown
# Gratuity

Gratuity is a lump-sum payment made by the employer to an employee as a token of appreciation for service.

## Eligibility
- Minimum 5 years of continuous service (4 years 240 days counts)
- Applicable to organizations with 10+ employees

## Calculation
Gratuity = (Last drawn Basic Pay × 15 × Years of service) / 26

Example:
- Basic Pay: ₹30,000/month
- Service: 7 years
- Gratuity = (30,000 × 15 × 7) / 26 = ₹1,21,154

## Tax Treatment
- Gratuity up to ₹20,00,000 is tax-exempt for private sector employees
- Government employees: fully exempt

## In CTC
- Employers often include gratuity as part of CTC
- It's calculated as approximately 4.81% of Basic Pay per year
- This is a notional cost — you only receive it after 5 years
```

Create `knowledge_base/professional_tax.md`:

```markdown
# Professional Tax

Professional Tax is a state-level tax on employment income, deducted monthly by the employer.

## Key Facts
- Maximum ₹2,500 per year (as per Constitutional limit)
- Varies by state — not all states levy it
- Deducted monthly from salary

## State-wise Rates (Common Examples)
- **Karnataka**: ₹200/month (salary > ₹15,000)
- **Maharashtra**: ₹200/month (salary > ₹10,000), ₹300 in February
- **West Bengal**: ₹150-200/month based on slabs
- **Tamil Nadu**: ₹150/month (salary > ₹21,000)
- **Andhra Pradesh/Telangana**: ₹200/month (salary > ₹20,000)

## Tax Benefit
- Professional Tax paid is deductible under Section 16(iii) of Income Tax Act
- Reduces your taxable income
```

Create `knowledge_base/offer_letter_terms.md`:

```markdown
# Offer Letter Terms

An offer letter is a formal document from an employer to a prospective employee specifying the terms of employment.

## Key Components
- **CTC**: Total cost to company (annual)
- **Fixed Pay**: Guaranteed monthly salary components
- **Variable Pay**: Performance-linked bonus (quarterly/annual)
- **Joining Bonus**: One-time payment on joining (may have clawback clause)
- **Probation Period**: Trial period (typically 3-6 months) with notice period often shorter
- **Notice Period**: Time required before resignation takes effect (30-90 days)
- **ESOP/RSU**: Stock options or restricted stock units (vesting schedule matters)

## Things to Check
- Is variable pay guaranteed or performance-based?
- What's the clawback clause on joining bonus?
- What happens during probation — is PF/insurance included?
- Notice period during vs. after probation
- Does CTC include employer PF and gratuity? (It usually does)
- Relocation allowance terms

## Red Flags
- Very high variable component (>30% of CTC)
- Long notice period with no buy-out option
- Joining bonus with >1 year clawback
- No mention of PF or insurance
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_knowledge_base.py`:

```python
"""Tests for payroll knowledge base loading."""

import os
import shutil
import pytest
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.knowledge_base import load_knowledge_base


TEST_CHROMA_DIR = "./test_chroma_kb"
KB_DIR = "./knowledge_base"


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel("all-MiniLM-L6-v2")


@pytest.fixture
def kb_vector_store(embedding_model):
    store = VectorStore(
        collection_name="test_kb",
        persist_dir=TEST_CHROMA_DIR,
        embedding_model=embedding_model,
    )
    yield store
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR)


def test_load_knowledge_base_indexes_documents(kb_vector_store):
    """Should load all KB markdown files and index chunks."""
    count = load_knowledge_base(kb_vector_store, KB_DIR)
    assert count > 0
    assert kb_vector_store.count() > 0


def test_load_knowledge_base_returns_chunk_count(kb_vector_store):
    """Should return the number of chunks indexed."""
    count = load_knowledge_base(kb_vector_store, KB_DIR)
    assert isinstance(count, int)
    assert count == kb_vector_store.count()


def test_kb_query_finds_relevant_content(kb_vector_store):
    """After loading KB, queries should return relevant payroll info."""
    load_knowledge_base(kb_vector_store, KB_DIR)
    results = kb_vector_store.query("What is HRA and how is it calculated?", n_results=3)
    assert len(results) > 0
    # At least one result should mention HRA
    texts = " ".join(r["text"] for r in results)
    assert "HRA" in texts or "House Rent" in texts
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_knowledge_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write knowledge base loader implementation**

Create `src/rag/knowledge_base.py`:

```python
"""Load and index the payroll knowledge base into the vector store."""

import os
from src.rag.vector_store import VectorStore


def _chunk_markdown(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split markdown text into overlapping chunks by paragraphs.

    Splits on double newlines (paragraph boundaries), then merges small
    paragraphs and splits large ones to stay near chunk_size.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If a single paragraph exceeds chunk_size, split it by sentences
            if len(para) > chunk_size:
                words = para.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 > chunk_size:
                        chunks.append(current_chunk.strip())
                        # Keep overlap
                        overlap_words = current_chunk.strip().split()[-10:]
                        current_chunk = " ".join(overlap_words) + " " + word
                    else:
                        current_chunk = (current_chunk + " " + word).strip()
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def load_knowledge_base(vector_store: VectorStore, kb_dir: str) -> int:
    """
    Load all markdown files from the knowledge base directory into the vector store.

    Args:
        vector_store: The VectorStore to index into.
        kb_dir: Path to the knowledge_base directory containing .md files.

    Returns:
        Total number of chunks indexed.
    """
    if not os.path.isdir(kb_dir):
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    total_chunks = 0

    for filename in sorted(os.listdir(kb_dir)):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(kb_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        topic = filename.replace(".md", "")
        chunks = _chunk_markdown(content)

        metadatas = [
            {"source": "knowledge_base", "topic": topic, "filename": filename}
            for _ in chunks
        ]

        vector_store.add_documents(
            doc_id=f"kb_{topic}",
            chunks=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    return total_chunks
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_knowledge_base.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add knowledge_base/ src/rag/knowledge_base.py tests/test_knowledge_base.py
git commit -m "feat: add payroll knowledge base with 9 topics and KB loader"
```

---

## Task 7: Retriever

**Files:**
- Create: `paysense-ai/src/rag/retriever.py`
- Test: `paysense-ai/tests/test_retriever.py`

**Interfaces:**
- Consumes: `VectorStore` (two instances — one for KB, one for user docs)
- Produces: `Retriever(kb_store: VectorStore, user_store: VectorStore)` with method `retrieve(query: str, n_results: int = 5) -> list[dict]` — merges results from both stores, deduplicates, sorts by relevance (distance).

- [ ] **Step 1: Write the failing test**

Create `tests/test_retriever.py`:

```python
"""Tests for the combined retriever (KB + user docs)."""

import os
import shutil
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
    kb_store = VectorStore("test_kb", TEST_CHROMA_DIR, embedding_model)
    user_store = VectorStore("test_user", TEST_CHROMA_DIR, embedding_model)

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

    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/rag/retriever.py`:

```python
"""Combined retriever that searches both knowledge base and user document stores."""

from src.rag.vector_store import VectorStore


class Retriever:
    """Retrieves relevant context from both KB and user document vector stores."""

    def __init__(self, kb_store: VectorStore, user_store: VectorStore):
        self._kb_store = kb_store
        self._user_store = user_store

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Search both stores and return merged, deduplicated, sorted results.

        Args:
            query: The user's question or search query.
            n_results: Maximum total results to return.

        Returns:
            List of dicts sorted by distance (ascending = most relevant first).
            Each dict has keys: text, metadata, distance.
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
            text = result["text"]
            if text not in seen_texts:
                seen_texts.add(text)
                merged.append(result)

        # Sort by distance (lower = more relevant)
        merged.sort(key=lambda r: r["distance"])

        return merged[:n_results]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retriever.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retriever.py tests/test_retriever.py
git commit -m "feat: add combined retriever for KB and user document search"
```

---

## Task 8: Document Comparator

**Files:**
- Create: `paysense-ai/src/comparison/__init__.py`
- Create: `paysense-ai/src/comparison/comparator.py`
- Test: `paysense-ai/tests/test_comparator.py`

**Interfaces:**
- Consumes: Two field dictionaries (output of `extract_fields` from Task 4)
- Produces: `compare_documents(doc_a: dict, doc_b: dict) -> dict` — returns `{"summary": str, "changes": list[dict], "doc_a_label": str, "doc_b_label": str}`. Each change dict: `{"field": str, "label": str, "value_a": any, "value_b": any, "change": str, "explanation": str}`.

- [ ] **Step 1: Create `src/comparison/__init__.py`**

```python
"""Document comparison logic."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_comparator.py`:

```python
"""Tests for document comparison logic."""

import pytest
from src.comparison.comparator import compare_documents


DOC_A = {
    "document_type": "payslip",
    "employee_name": "Rahul Sharma",
    "company_name": "TechCorp",
    "period": "January 2025",
    "basic_pay.md": 25000,
    "hra": 12500,
    "da": 5000,
    "pf_employee": 3000,
    "pf_employer": 3000,
    "esi": None,
    "tds": 4167,
    "professional_tax": 200,
    "gross_salary": 50000,
    "net_pay": 42258,
    "ctc": None,
    "other_allowances": {"special_allowance": 7500},
    "other_deductions": {},
}

DOC_B = {
    "document_type": "payslip",
    "employee_name": "Rahul Sharma",
    "company_name": "TechCorp",
    "period": "June 2025",
    "basic_pay.md": 28000,
    "hra": 14000,
    "da": 5500,
    "pf_employee": 3360,
    "pf_employer": 3360,
    "esi": None,
    "tds": 5000,
    "professional_tax": 200,
    "gross_salary": 56000,
    "net_pay": 47040,
    "ctc": None,
    "other_allowances": {"special_allowance": 8500},
    "other_deductions": {},
}


def test_compare_documents_returns_expected_structure():
    """Should return a dict with summary, changes, and labels."""
    result = compare_documents(DOC_A, DOC_B)
    assert "summary" in result
    assert "changes" in result
    assert "doc_a_label" in result
    assert "doc_b_label" in result
    assert isinstance(result["changes"], list)


def test_compare_documents_detects_changes():
    """Should detect fields that changed between documents."""
    result = compare_documents(DOC_A, DOC_B)
    changed_fields = {c["field"] for c in result["changes"]}
    assert "basic_pay.md" in changed_fields
    assert "hra" in changed_fields
    assert "net_pay" in changed_fields


def test_compare_documents_shows_values():
    """Each change should have value_a, value_b, and change description."""
    result = compare_documents(DOC_A, DOC_B)
    for change in result["changes"]:
        assert "field" in change
        assert "value_a" in change
        assert "value_b" in change
        assert "change" in change


def test_compare_documents_unchanged_not_included():
    """Fields that didn't change should not appear in changes."""
    result = compare_documents(DOC_A, DOC_B)
    changed_fields = {c["field"] for c in result["changes"]}
    # professional_tax is 200 in both — should NOT appear
    assert "professional_tax" not in changed_fields


def test_compare_documents_labels_use_period():
    """Labels should use the document period if available."""
    result = compare_documents(DOC_A, DOC_B)
    assert "January 2025" in result["doc_a_label"]
    assert "June 2025" in result["doc_b_label"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_comparator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write minimal implementation**

Create `src/comparison/comparator.py`:

```python
"""Compare two payslip/offer letter documents field-by-field."""

# Fields to compare (in display order) with human-readable labels
COMPARABLE_FIELDS = [
    ("basic_pay.md", "Basic Pay"),
    ("hra", "HRA"),
    ("da", "Dearness Allowance"),
    ("pf_employee", "PF (Employee)"),
    ("pf_employer", "PF (Employer)"),
    ("esi", "ESI"),
    ("tds", "TDS / Income Tax"),
    ("professional_tax", "Professional Tax"),
    ("gross_salary", "Gross Salary"),
    ("net_pay", "Net Pay / In-Hand"),
    ("ctc", "CTC"),
]


def _format_change(old_val, new_val) -> str:
    """Describe how a numeric value changed."""
    if old_val is None and new_val is None:
        return "No change"
    if old_val is None:
        return f"New: ₹{new_val:,.0f}"
    if new_val is None:
        return f"Removed (was ₹{old_val:,.0f})"

    diff = new_val - old_val
    if diff == 0:
        return "No change"

    pct = (diff / old_val * 100) if old_val != 0 else 0
    direction = "increased" if diff > 0 else "decreased"
    return f"{direction} by ₹{abs(diff):,.0f} ({abs(pct):.1f}%)"


def _make_label(doc: dict) -> str:
    """Create a label for the document using period and type."""
    parts = []
    if doc.get("document_type") and doc["document_type"] != "unknown":
        parts.append(doc["document_type"].replace("_", " ").title())
    if doc.get("period"):
        parts.append(f"({doc['period']})")
    return " ".join(parts) if parts else "Document"


def compare_documents(doc_a: dict, doc_b: dict) -> dict:
    """
    Compare two extracted document field dicts and return a structured diff.

    Args:
        doc_a: First document's extracted fields.
        doc_b: Second document's extracted fields.

    Returns:
        Dict with keys:
        - summary: A short text summary of the comparison
        - changes: List of dicts for each changed field
        - doc_a_label: Human-readable label for doc_a
        - doc_b_label: Human-readable label for doc_b
    """
    changes = []

    for field, label in COMPARABLE_FIELDS:
        val_a = doc_a.get(field)
        val_b = doc_b.get(field)

        # Skip if both are None
        if val_a is None and val_b is None:
            continue

        # Skip if no change
        if val_a == val_b:
            continue

        changes.append(
            {
                "field": field,
                "label": label,
                "value_a": val_a,
                "value_b": val_b,
                "change": _format_change(val_a, val_b),
            }
        )

    # Compare other_allowances
    other_a = doc_a.get("other_allowances", {}) or {}
    other_b = doc_b.get("other_allowances", {}) or {}
    all_other_keys = set(list(other_a.keys()) + list(other_b.keys()))
    for key in sorted(all_other_keys):
        va = other_a.get(key)
        vb = other_b.get(key)
        if va != vb:
            changes.append(
                {
                    "field": f"other_allowances.{key}",
                    "label": key.replace("_", " ").title(),
                    "value_a": va,
                    "value_b": vb,
                    "change": _format_change(va, vb) if isinstance(va, (int, float)) or isinstance(vb, (int, float)) else f"{va} → {vb}",
                }
            )

    doc_a_label = _make_label(doc_a)
    doc_b_label = _make_label(doc_b)

    # Summary
    num_increased = sum(1 for c in changes if "increased" in c["change"])
    num_decreased = sum(1 for c in changes if "decreased" in c["change"])
    summary = f"{len(changes)} field(s) changed: {num_increased} increased, {num_decreased} decreased."

    return {
        "summary": summary,
        "changes": changes,
        "doc_a_label": doc_a_label,
        "doc_b_label": doc_b_label,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_comparator.py -v`
Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/comparison/ tests/test_comparator.py
git commit -m "feat: add field-by-field document comparator with change explanations"
```

---

## Task 9: Session Manager

**Files:**
- Create: `paysense-ai/src/sessions/__init__.py`
- Create: `paysense-ai/src/sessions/session_manager.py`
- Test: `paysense-ai/tests/test_session_manager.py`

**Interfaces:**
- Consumes: Nothing (standalone)
- Produces: `SessionManager()` with methods:
  - `create_session() -> str` — returns a new session ID (UUID)
  - `get_session(session_id: str) -> dict` — returns session data dict. Raises `KeyError` if not found.
  - `add_document(session_id: str, doc_id: str, raw_text: str, fields: dict) -> None`
  - `get_documents(session_id: str) -> dict[str, dict]` — returns `{doc_id: {"raw_text": str, "fields": dict}}`
  - `add_message(session_id: str, role: str, content: str) -> None`
  - `get_messages(session_id: str) -> list[dict]` — returns list of `{"role": str, "content": str}`
  - `delete_session(session_id: str) -> None`

- [ ] **Step 1: Create `src/sessions/__init__.py`**

```python
"""Session management for multi-user support."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_session_manager.py`:

```python
"""Tests for in-memory session manager."""

import pytest
from src.sessions.session_manager import SessionManager


@pytest.fixture
def manager():
    return SessionManager()


def test_create_session_returns_uuid(manager):
    """Should return a valid UUID string."""
    sid = manager.create_session()
    assert isinstance(sid, str)
    assert len(sid) == 36  # UUID format: 8-4-4-4-12


def test_get_session_returns_dict(manager):
    """Should return session data dict."""
    sid = manager.create_session()
    data = manager.get_session(sid)
    assert isinstance(data, dict)
    assert "documents" in data
    assert "messages" in data


def test_get_session_invalid_id_raises(manager):
    """Should raise KeyError for unknown session."""
    with pytest.raises(KeyError):
        manager.get_session("nonexistent-id")


def test_add_and_get_document(manager):
    """Should store and retrieve documents by session."""
    sid = manager.create_session()
    manager.add_document(sid, "doc1", "raw text here", {"basic_pay.md": 25000})
    docs = manager.get_documents(sid)
    assert "doc1" in docs
    assert docs["doc1"]["raw_text"] == "raw text here"
    assert docs["doc1"]["fields"]["basic_pay.md"] == 25000


def test_add_and_get_messages(manager):
    """Should store and retrieve chat messages in order."""
    sid = manager.create_session()
    manager.add_message(sid, "user", "What is my salary?")
    manager.add_message(sid, "assistant", "Your net pay is ₹42,258.")
    messages = manager.get_messages(sid)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_delete_session(manager):
    """Should remove session completely."""
    sid = manager.create_session()
    manager.delete_session(sid)
    with pytest.raises(KeyError):
        manager.get_session(sid)


def test_sessions_are_isolated(manager):
    """Documents in one session should not appear in another."""
    sid1 = manager.create_session()
    sid2 = manager.create_session()
    manager.add_document(sid1, "doc1", "text1", {"basic_pay.md": 100})
    docs1 = manager.get_documents(sid1)
    docs2 = manager.get_documents(sid2)
    assert "doc1" in docs1
    assert "doc1" not in docs2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_session_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write minimal implementation**

Create `src/sessions/session_manager.py`:

```python
"""In-memory session store for multi-user isolation."""

import uuid
from threading import Lock


class SessionManager:
    """Thread-safe in-memory session manager."""

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

    def add_document(
        self, session_id: str, doc_id: str, raw_text: str, fields: dict
    ) -> None:
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_session_manager.py -v`
Expected: 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/sessions/ tests/test_session_manager.py
git commit -m "feat: add in-memory session manager for multi-user isolation"
```

---

## Task 10: Agent — Prompts, Tools, and Agent Loop

**Files:**
- Create: `paysense-ai/src/agent/__init__.py`
- Create: `paysense-ai/src/agent/prompts.py`
- Create: `paysense-ai/src/agent/tools.py`
- Create: `paysense-ai/src/agent/agent.py`
- Test: `paysense-ai/tests/test_agent.py`

**Interfaces:**
- Consumes: `Retriever` (Task 7), `SessionManager` (Task 9), `compare_documents` (Task 8), `Settings` (Task 1)
- Produces: `PaySenseAgent(retriever: Retriever, session_manager: SessionManager, settings: Settings)` with method `chat(session_id: str, user_message: str) -> str` — returns the agent's response as a string.

- [ ] **Step 1: Create `src/agent/__init__.py`**

```python
"""AI agent with RAG-powered tools for payslip and offer letter analysis."""
```

- [ ] **Step 2: Create `src/agent/prompts.py`**

```python
"""System prompts and prompt templates for the PaySense agent."""

SYSTEM_PROMPT = """You are PaySense AI, a helpful assistant that specializes in explaining Indian payslips and offer letters.

Your capabilities:
1. **Answer questions** about payslip components (Basic Pay, HRA, PF, TDS, etc.)
2. **Explain documents** that users have uploaded
3. **Compare documents** side-by-side and explain differences

Rules:
- Always cite the source of your information (knowledge base topic or uploaded document)
- When explaining monetary values, use ₹ symbol and Indian numbering (e.g., ₹1,25,000)
- If you don't know something or can't find it in the context, say so clearly
- Be concise but thorough — explain payroll terms in simple language
- When comparing documents, highlight the most impactful changes first
- Never make up or calculate values that aren't in the provided context

You have access to the following tools:
- search_knowledge: Search the payroll knowledge base for explanations of payroll concepts
- search_documents: Search the user's uploaded documents for specific information
- compare_docs: Compare two uploaded documents field-by-field
"""

QA_PROMPT_TEMPLATE = """Context from knowledge base and uploaded documents:
{context}

User's uploaded document fields (if any):
{document_fields}

Chat history:
{chat_history}

User question: {question}

Provide a clear, helpful answer. Cite sources when referencing specific information."""

COMPARE_PROMPT_TEMPLATE = """You are comparing two documents for the user.

Comparison results:
{comparison_data}

Document A: {doc_a_label}
Document B: {doc_b_label}

Explain the differences in simple language. Highlight:
1. The most significant changes (by amount and percentage)
2. What the changes mean for the employee (e.g., impact on take-home, tax, retirement)
3. Any unusual or potentially concerning changes

User's question: {question}"""
```

- [ ] **Step 3: Create `src/agent/tools.py`**

```python
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


def compare_docs(
    session_manager: SessionManager, session_id: str, doc_id_a: str, doc_id_b: str
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
```

- [ ] **Step 4: Create `src/agent/agent.py`**

```python
"""PaySense AI agent — orchestrates RAG retrieval, tools, and LLM responses."""

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
                comparison = compare_docs(
                    self._session_manager, session_id, doc_ids[0], doc_ids[1]
                )
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
```

- [ ] **Step 5: Write agent tests (mocking LLM calls)**

Create `tests/test_agent.py`:

```python
"""Tests for the PaySense AI agent."""

import os
import shutil
import pytest
from unittest.mock import patch
from src.config import Settings
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.sessions.session_manager import SessionManager
from src.agent.agent import PaySenseAgent


TEST_CHROMA_DIR = "./test_chroma_agent"


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel("all-MiniLM-L6-v2")


@pytest.fixture
def agent_setup(embedding_model):
    """Set up the full agent with seeded stores."""
    kb_store = VectorStore("test_kb_agent", TEST_CHROMA_DIR, embedding_model)
    user_store = VectorStore("test_user_agent", TEST_CHROMA_DIR, embedding_model)

    # Seed KB
    kb_store.add_documents(
        "kb_hra",
        ["HRA stands for House Rent Allowance. It covers accommodation expenses."],
        [{"source": "knowledge_base", "topic": "hra"}],
    )

    retriever = Retriever(kb_store, user_store)
    session_mgr = SessionManager()
    settings = Settings(gemini_api_key="fake-key", chroma_persist_dir=TEST_CHROMA_DIR)

    agent = PaySenseAgent(retriever, session_mgr, settings)
    session_id = session_mgr.create_session()

    yield agent, session_mgr, session_id

    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR)


@patch("src.agent.agent.PaySenseAgent._call_llm")
def test_chat_returns_response(mock_llm, agent_setup):
    """Agent should return a string response."""
    mock_llm.return_value = "HRA stands for House Rent Allowance."
    agent, session_mgr, session_id = agent_setup
    response = agent.chat(session_id, "What is HRA?")
    assert isinstance(response, str)
    assert len(response) > 0


@patch("src.agent.agent.PaySenseAgent._call_llm")
def test_chat_stores_messages(mock_llm, agent_setup):
    """Agent should store both user and assistant messages."""
    mock_llm.return_value = "Your net pay is ₹42,258."
    agent, session_mgr, session_id = agent_setup
    agent.chat(session_id, "What is my net pay?")
    messages = session_mgr.get_messages(session_id)
    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles


@patch("src.agent.agent.PaySenseAgent._call_llm")
def test_chat_comparison_query(mock_llm, agent_setup):
    """Agent should detect comparison queries and use comparison logic."""
    mock_llm.return_value = "Your basic pay increased by 12%."
    agent, session_mgr, session_id = agent_setup

    # Add two documents to session
    session_mgr.add_document(session_id, "jan_payslip", "text", {
        "document_type": "payslip", "period": "Jan 2025",
        "basic_pay.md": 25000, "net_pay": 42000,
    })
    session_mgr.add_document(session_id, "jun_payslip", "text", {
        "document_type": "payslip", "period": "Jun 2025",
        "basic_pay.md": 28000, "net_pay": 47000,
    })

    response = agent.chat(session_id, "Compare my two payslips")
    assert isinstance(response, str)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_agent.py -v`
Expected: 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/agent/ tests/test_agent.py
git commit -m "feat: add PaySense AI agent with RAG tools and comparison support"
```

---

## Task 11: FastAPI Backend — Models, Upload, Chat, Compare Endpoints

**Files:**
- Create: `paysense-ai/src/api/__init__.py`
- Create: `paysense-ai/src/api/models.py`
- Create: `paysense-ai/src/api/routes_upload.py`
- Create: `paysense-ai/src/api/routes_chat.py`
- Create: `paysense-ai/src/api/routes_compare.py`
- Create: `paysense-ai/src/api/app.py`
- Create: `paysense-ai/run.py`
- Test: `paysense-ai/tests/test_api_upload.py`
- Test: `paysense-ai/tests/test_api_chat.py`
- Test: `paysense-ai/tests/test_api_compare.py`

**Interfaces:**
- Consumes: `PaySenseAgent` (Task 10), `SessionManager` (Task 9), parsers (Tasks 2-4), `Settings` (Task 1)
- Produces: FastAPI app with endpoints:
  - `POST /api/session` → `{"session_id": str}`
  - `POST /api/upload` → `{"doc_id": str, "fields": dict, "message": str}`
  - `POST /api/chat` → `{"response": str}`
  - `POST /api/compare` → `{"comparison": dict, "explanation": str}`
  - `GET /api/session/{session_id}/documents` → `{"documents": dict}`

- [ ] **Step 1: Create `src/api/__init__.py`**

```python
"""FastAPI REST API for PaySense AI."""
```

- [ ] **Step 2: Create `src/api/models.py`**

```python
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
```

- [ ] **Step 3: Create `src/api/routes_upload.py`**

```python
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
```

- [ ] **Step 4: Create `src/api/routes_chat.py`**

```python
"""Chat endpoint — handles user questions via the RAG agent."""

from fastapi import APIRouter, HTTPException
from src.api.models import ChatRequest, ChatResponse

router = APIRouter()

_agent = None


def init_chat_routes(agent):
    """Inject the PaySense agent dependency."""
    global _agent
    _agent = agent


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the PaySense AI agent."""
    try:
        response = _agent.chat(request.session_id, request.message)
        return ChatResponse(response=response)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
```

- [ ] **Step 5: Create `src/api/routes_compare.py`**

```python
"""Compare endpoint — handles document comparison requests."""

import json
from fastapi import APIRouter, HTTPException
from src.api.models import CompareRequest, CompareResponse
from src.comparison.comparator import compare_documents

router = APIRouter()

_session_manager = None
_agent = None


def init_compare_routes(session_manager, agent):
    """Inject dependencies for comparison routes."""
    global _session_manager, _agent
    _session_manager = session_manager
    _agent = agent


@router.post("/api/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """Compare two uploaded documents."""
    try:
        docs = _session_manager.get_documents(request.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.doc_id_a not in docs:
        raise HTTPException(status_code=404, detail=f"Document '{request.doc_id_a}' not found")
    if request.doc_id_b not in docs:
        raise HTTPException(status_code=404, detail=f"Document '{request.doc_id_b}' not found")

    fields_a = docs[request.doc_id_a]["fields"]
    fields_b = docs[request.doc_id_b]["fields"]
    comparison = compare_documents(fields_a, fields_b)

    # Get LLM explanation via the agent
    explanation = _agent.chat(
        request.session_id,
        f"Compare documents: {json.dumps(comparison, default=str)}. User asked: {request.question}",
    )

    return CompareResponse(comparison=comparison, explanation=explanation)
```

- [ ] **Step 6: Create `src/api/app.py`**

```python
"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.models import SessionResponse
from src.api.routes_upload import router as upload_router, init_upload_routes
from src.api.routes_chat import router as chat_router, init_chat_routes
from src.api.routes_compare import router as compare_router, init_compare_routes
from src.config import Settings
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.knowledge_base import load_knowledge_base
from src.rag.retriever import Retriever
from src.sessions.session_manager import SessionManager
from src.agent.agent import PaySenseAgent


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings.from_env()

    app = FastAPI(
        title="PaySense AI",
        description="Smart Payslip & Offer Letter Assistant powered by RAG",
        version="1.0.0",
    )

    # Initialize components
    embedding_model = EmbeddingModel(settings.embedding_model_name)
    kb_store = VectorStore("payroll_kb", settings.chroma_persist_dir, embedding_model)
    user_store = VectorStore("user_documents", settings.chroma_persist_dir, embedding_model)

    # Load knowledge base (skip if already loaded)
    if kb_store.count() == 0:
        load_knowledge_base(kb_store, "./knowledge_base")

    retriever = Retriever(kb_store, user_store)
    session_manager = SessionManager()
    agent = PaySenseAgent(retriever, session_manager, settings)

    # Inject dependencies
    init_upload_routes(session_manager, user_store, settings)
    init_chat_routes(agent)
    init_compare_routes(session_manager, agent)

    # Register routers
    app.include_router(upload_router)
    app.include_router(chat_router)
    app.include_router(compare_router)

    # Session creation endpoint
    @app.post("/api/session", response_model=SessionResponse)
    async def create_session():
        session_id = session_manager.create_session()
        return SessionResponse(session_id=session_id)

    # Document list endpoint
    @app.get("/api/session/{session_id}/documents")
    async def get_documents(session_id: str):
        try:
            docs = session_manager.get_documents(session_id)
            # Return just fields (not raw text) for the frontend
            return {
                "documents": {
                    doc_id: {"fields": data["fields"]}
                    for doc_id, data in docs.items()
                }
            }
        except KeyError:
            return {"error": "Session not found"}

    # Serve frontend
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app
```

- [ ] **Step 7: Create `run.py`**

```python
"""Entry point — run the PaySense AI server."""

import uvicorn
from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 8: Write API tests**

Create `tests/test_api_upload.py`:

```python
"""Tests for the upload API endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config import Settings


@pytest.fixture
def client():
    settings = Settings(gemini_api_key="test-key", chroma_persist_dir="./test_chroma_api")
    with patch("src.api.app.EmbeddingModel"):
        with patch("src.api.app.VectorStore") as mock_vs:
            mock_vs.return_value.count.return_value = 1
            with patch("src.api.app.load_knowledge_base"):
                app = create_app(settings)
                yield TestClient(app)


def test_create_session(client):
    """POST /api/session should return a session_id."""
    response = client.post("/api/session")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 36
```

Create `tests/test_api_chat.py`:

```python
"""Tests for the chat API endpoint."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config import Settings


@pytest.fixture
def client():
    settings = Settings(gemini_api_key="test-key", chroma_persist_dir="./test_chroma_api_chat")
    with patch("src.api.app.EmbeddingModel"):
        with patch("src.api.app.VectorStore") as mock_vs:
            mock_vs.return_value.count.return_value = 1
            with patch("src.api.app.load_knowledge_base"):
                with patch("src.agent.agent.PaySenseAgent._call_llm", return_value="Test response"):
                    app = create_app(settings)
                    yield TestClient(app)


def test_chat_endpoint(client):
    """POST /api/chat should return agent response."""
    # Create session first
    session_resp = client.post("/api/session")
    session_id = session_resp.json()["session_id"]

    # Send chat message
    response = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "What is HRA?"
    })
    assert response.status_code == 200
    assert "response" in response.json()


def test_chat_invalid_session(client):
    """Should return 404 for invalid session."""
    response = client.post("/api/chat", json={
        "session_id": "invalid-session-id",
        "message": "Hello"
    })
    assert response.status_code == 404
```

Create `tests/test_api_compare.py`:

```python
"""Tests for the compare API endpoint."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config import Settings


@pytest.fixture
def client():
    settings = Settings(gemini_api_key="test-key", chroma_persist_dir="./test_chroma_api_cmp")
    with patch("src.api.app.EmbeddingModel"):
        with patch("src.api.app.VectorStore") as mock_vs:
            mock_vs.return_value.count.return_value = 1
            with patch("src.api.app.load_knowledge_base"):
                with patch("src.agent.agent.PaySenseAgent._call_llm", return_value="Comparison explanation"):
                    app = create_app(settings)
                    yield TestClient(app)


def test_compare_missing_documents(client):
    """Should return 404 when documents don't exist."""
    session_resp = client.post("/api/session")
    session_id = session_resp.json()["session_id"]

    response = client.post("/api/compare", json={
        "session_id": session_id,
        "doc_id_a": "nonexistent_a",
        "doc_id_b": "nonexistent_b",
    })
    assert response.status_code == 404
```

- [ ] **Step 9: Run all API tests**

Run: `python -m pytest tests/test_api_upload.py tests/test_api_chat.py tests/test_api_compare.py -v`
Expected: 4 tests PASS

- [ ] **Step 10: Commit**

```bash
git add src/api/ run.py tests/test_api_upload.py tests/test_api_chat.py tests/test_api_compare.py
git commit -m "feat: add FastAPI backend with upload, chat, and compare endpoints"
```

---

## Task 12: Frontend — Upload, Chat, and Compare UI

**Files:**
- Create: `paysense-ai/frontend/index.html`
- Create: `paysense-ai/frontend/style.css`
- Create: `paysense-ai/frontend/app.js`

**Interfaces:**
- Consumes: All API endpoints from Task 11
- Produces: Single-page web app with three panels: Upload, Chat, Comparison

- [ ] **Step 1: Create `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="PaySense AI - Smart Payslip & Offer Letter Assistant powered by RAG. Upload, understand, and compare your salary documents.">
    <title>PaySense AI — Smart Payslip & Offer Letter Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="app-header">
            <div class="logo">
                <span class="logo-icon">💰</span>
                <h1>PaySense <span class="ai-badge">AI</span></h1>
            </div>
            <p class="tagline">Upload your payslip or offer letter. Ask anything.</p>
        </header>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Left Panel: Upload & Documents -->
            <section class="panel upload-panel" id="upload-panel">
                <div class="panel-header">
                    <h2>📄 Documents</h2>
                    <span class="doc-count" id="doc-count">0 uploaded</span>
                </div>

                <div class="upload-zone" id="upload-zone">
                    <div class="upload-icon">⬆️</div>
                    <p>Drop your payslip or offer letter here</p>
                    <p class="upload-hint">PDF, PNG, JPG — max 10MB</p>
                    <input type="file" id="file-input" accept=".pdf,.png,.jpg,.jpeg" hidden>
                    <button class="btn btn-primary" id="browse-btn">Browse Files</button>
                </div>

                <div class="upload-progress hidden" id="upload-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <p class="progress-text" id="progress-text">Processing...</p>
                </div>

                <div class="document-list" id="document-list">
                    <!-- Documents appear here -->
                </div>

                <div class="compare-section hidden" id="compare-section">
                    <h3>📊 Compare Documents</h3>
                    <select id="compare-doc-a" class="select-doc"></select>
                    <span class="vs-label">vs</span>
                    <select id="compare-doc-b" class="select-doc"></select>
                    <button class="btn btn-secondary" id="compare-btn">Compare</button>
                </div>
            </section>

            <!-- Right Panel: Chat -->
            <section class="panel chat-panel" id="chat-panel">
                <div class="panel-header">
                    <h2>💬 Ask PaySense</h2>
                </div>

                <div class="chat-messages" id="chat-messages">
                    <div class="message assistant-message">
                        <div class="message-avatar">🤖</div>
                        <div class="message-content">
                            <p>Hi! I'm PaySense AI. Upload a payslip or offer letter, then ask me anything about it.</p>
                            <p class="message-hint">Try: "What is my in-hand salary?", "Explain my HRA", "How much tax am I paying?"</p>
                        </div>
                    </div>
                </div>

                <div class="chat-input-area">
                    <textarea id="chat-input" placeholder="Ask about your payslip or offer letter..." rows="1"></textarea>
                    <button class="btn btn-send" id="send-btn" disabled>
                        <span>➤</span>
                    </button>
                </div>
            </section>
        </main>
    </div>

    <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `frontend/style.css`**

```css
/* ============================================
   PaySense AI — Design System & Styles
   ============================================ */

:root {
    /* Color Palette */
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a1f35;
    --bg-card-hover: #222842;
    --bg-input: #151b2e;

    --text-primary: #e8eaf0;
    --text-secondary: #8b92a8;
    --text-muted: #5a6178;

    --accent-primary: #6366f1;
    --accent-primary-hover: #818cf8;
    --accent-gradient: linear-gradient(135deg, #6366f1, #8b5cf6);
    --accent-glow: rgba(99, 102, 241, 0.15);

    --success: #34d399;
    --warning: #fbbf24;
    --error: #f87171;

    --border-color: rgba(255, 255, 255, 0.06);
    --border-active: rgba(99, 102, 241, 0.4);

    /* Spacing */
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;

    /* Border Radius */
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-full: 9999px;

    /* Shadows */
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
    --shadow-glow: 0 0 20px var(--accent-glow);

    /* Transitions */
    --transition-fast: 150ms ease;
    --transition-normal: 250ms ease;
    --transition-slow: 400ms cubic-bezier(0.4, 0, 0.2, 1);
}

/* ============ Reset & Base ============ */

*, *::before, *::after {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* ============ App Container ============ */

.app-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: var(--spacing-lg);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* ============ Header ============ */

.app-header {
    text-align: center;
    padding: var(--spacing-xl) 0 var(--spacing-lg);
    animation: fadeInDown 0.6s ease;
}

.logo {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--spacing-sm);
}

.logo-icon {
    font-size: 2rem;
    animation: pulse 2s ease-in-out infinite;
}

.logo h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.ai-badge {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.5rem;
    font-weight: 600;
}

.tagline {
    color: var(--text-secondary);
    margin-top: var(--spacing-xs);
    font-size: 0.95rem;
}

/* ============ Main Content ============ */

.main-content {
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: var(--spacing-lg);
    flex: 1;
    min-height: 0;
    animation: fadeIn 0.8s ease;
}

/* ============ Panels ============ */

.panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: var(--shadow-md);
    transition: border-color var(--transition-normal);
}

.panel:hover {
    border-color: var(--border-active);
}

.panel-header {
    padding: var(--spacing-lg);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-card);
}

.panel-header h2 {
    font-size: 1.1rem;
    font-weight: 600;
}

.doc-count {
    font-size: 0.8rem;
    color: var(--text-muted);
    background: var(--bg-primary);
    padding: 2px 10px;
    border-radius: var(--radius-full);
}

/* ============ Upload Zone ============ */

.upload-zone {
    margin: var(--spacing-lg);
    padding: var(--spacing-xl);
    border: 2px dashed var(--border-color);
    border-radius: var(--radius-md);
    text-align: center;
    cursor: pointer;
    transition: all var(--transition-normal);
    background: var(--bg-card);
}

.upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--accent-primary);
    background: var(--accent-glow);
    box-shadow: var(--shadow-glow);
}

.upload-icon {
    font-size: 2.5rem;
    margin-bottom: var(--spacing-sm);
}

.upload-zone p {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.upload-hint {
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
    margin-top: var(--spacing-xs);
}

/* ============ Buttons ============ */

.btn {
    padding: 10px 20px;
    border: none;
    border-radius: var(--radius-sm);
    font-family: inherit;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
}

.btn-primary {
    background: var(--accent-gradient);
    color: white;
    margin-top: var(--spacing-md);
    box-shadow: var(--shadow-sm);
}

.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-glow);
}

.btn-secondary {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background: var(--bg-card-hover);
    border-color: var(--accent-primary);
}

.btn-send {
    background: var(--accent-gradient);
    color: white;
    width: 44px;
    height: 44px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 1.2rem;
    padding: 0;
}

.btn-send:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.btn-send:not(:disabled):hover {
    transform: scale(1.05);
    box-shadow: var(--shadow-glow);
}

/* ============ Upload Progress ============ */

.upload-progress {
    margin: 0 var(--spacing-lg) var(--spacing-md);
    padding: var(--spacing-md);
    background: var(--bg-card);
    border-radius: var(--radius-sm);
}

.progress-bar {
    height: 4px;
    background: var(--bg-primary);
    border-radius: var(--radius-full);
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: var(--accent-gradient);
    border-radius: var(--radius-full);
    width: 0%;
    transition: width var(--transition-slow);
}

.progress-text {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: var(--spacing-xs);
}

/* ============ Document List ============ */

.document-list {
    padding: 0 var(--spacing-lg);
    flex: 1;
    overflow-y: auto;
}

.doc-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: var(--spacing-md);
    margin-bottom: var(--spacing-sm);
    transition: all var(--transition-fast);
    animation: slideInLeft 0.3s ease;
}

.doc-card:hover {
    border-color: var(--accent-primary);
    background: var(--bg-card-hover);
}

.doc-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-xs);
}

.doc-type-badge {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 2px 8px;
    border-radius: var(--radius-full);
    background: var(--accent-glow);
    color: var(--accent-primary-hover);
    font-weight: 600;
}

.doc-card-fields {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--spacing-xs);
    font-size: 0.8rem;
}

.doc-field {
    display: flex;
    justify-content: space-between;
    color: var(--text-secondary);
}

.doc-field-value {
    color: var(--text-primary);
    font-weight: 500;
}

/* ============ Compare Section ============ */

.compare-section {
    padding: var(--spacing-lg);
    border-top: 1px solid var(--border-color);
}

.compare-section h3 {
    font-size: 0.95rem;
    margin-bottom: var(--spacing-md);
}

.select-doc {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-family: inherit;
    font-size: 0.85rem;
    margin-bottom: var(--spacing-sm);
    cursor: pointer;
}

.select-doc:focus {
    outline: none;
    border-color: var(--accent-primary);
}

.vs-label {
    display: block;
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8rem;
    margin: var(--spacing-xs) 0;
}

#compare-btn {
    width: 100%;
    margin-top: var(--spacing-sm);
}

/* ============ Chat Panel ============ */

.chat-panel {
    max-height: calc(100vh - 160px);
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: var(--spacing-lg);
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.message {
    display: flex;
    gap: var(--spacing-md);
    animation: fadeIn 0.3s ease;
    max-width: 90%;
}

.user-message {
    align-self: flex-end;
    flex-direction: row-reverse;
}

.message-avatar {
    width: 36px;
    height: 36px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
    background: var(--bg-card);
}

.message-content {
    background: var(--bg-card);
    padding: var(--spacing-md);
    border-radius: var(--radius-md);
    font-size: 0.9rem;
    line-height: 1.7;
    border: 1px solid var(--border-color);
}

.user-message .message-content {
    background: var(--accent-glow);
    border-color: var(--border-active);
}

.message-content p + p {
    margin-top: var(--spacing-sm);
}

.message-hint {
    color: var(--text-muted);
    font-size: 0.8rem;
    font-style: italic;
}

/* ============ Chat Input ============ */

.chat-input-area {
    padding: var(--spacing-md) var(--spacing-lg);
    border-top: 1px solid var(--border-color);
    display: flex;
    gap: var(--spacing-sm);
    align-items: flex-end;
    background: var(--bg-card);
}

#chat-input {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    color: var(--text-primary);
    font-family: inherit;
    font-size: 0.9rem;
    resize: none;
    max-height: 120px;
    transition: border-color var(--transition-fast);
}

#chat-input:focus {
    outline: none;
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px var(--accent-glow);
}

#chat-input::placeholder {
    color: var(--text-muted);
}

/* ============ Utility Classes ============ */

.hidden {
    display: none !important;
}

/* ============ Animations ============ */

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading-spinner::after {
    content: '';
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid var(--text-muted);
    border-top-color: var(--accent-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-left: var(--spacing-sm);
}

/* ============ Scrollbar ============ */

::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: var(--radius-full);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* ============ Responsive ============ */

@media (max-width: 900px) {
    .main-content {
        grid-template-columns: 1fr;
    }

    .chat-panel {
        max-height: 60vh;
    }
}
```

- [ ] **Step 3: Create `frontend/app.js`**

```javascript
/**
 * PaySense AI — Frontend Application
 * Handles file upload, chat, and document comparison.
 */

const API_BASE = "";

// State
let sessionId = null;
const documents = {};

// DOM Elements
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const uploadZone = document.getElementById("upload-zone");
const uploadProgress = document.getElementById("upload-progress");
const progressFill = document.getElementById("progress-fill");
const progressText = document.getElementById("progress-text");
const documentList = document.getElementById("document-list");
const docCount = document.getElementById("doc-count");
const compareSection = document.getElementById("compare-section");
const compareDocA = document.getElementById("compare-doc-a");
const compareDocB = document.getElementById("compare-doc-b");
const compareBtn = document.getElementById("compare-btn");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

// ============ Session ============

async function initSession() {
    try {
        const resp = await fetch(`${API_BASE}/api/session`, { method: "POST" });
        const data = await resp.json();
        sessionId = data.session_id;
        console.log("Session created:", sessionId);
    } catch (err) {
        console.error("Failed to create session:", err);
        addMessage("assistant", "⚠️ Could not connect to the server. Please make sure the backend is running.");
    }
}

// ============ File Upload ============

browseBtn.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("click", (e) => {
    if (e.target !== browseBtn) fileInput.click();
});

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
    }
});

async function uploadFile(file) {
    if (!sessionId) {
        addMessage("assistant", "⚠️ Session not ready. Please wait...");
        return;
    }

    // Show progress
    uploadProgress.classList.remove("hidden");
    progressFill.style.width = "20%";
    progressText.textContent = "Uploading document...";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sessionId);

    try {
        progressFill.style.width = "50%";
        progressText.textContent = "Extracting text & fields...";

        const resp = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Upload failed");
        }

        progressFill.style.width = "100%";
        progressText.textContent = "Done!";

        const data = await resp.json();
        documents[data.doc_id] = data.fields;
        renderDocuments();
        addMessage("assistant", `✅ ${data.message}\n\nI found: **${data.fields.document_type}** for **${data.fields.employee_name || "Unknown"}** (${data.fields.period || "Unknown period"})`);

        setTimeout(() => {
            uploadProgress.classList.add("hidden");
            progressFill.style.width = "0%";
        }, 1500);
    } catch (err) {
        progressText.textContent = `❌ ${err.message}`;
        progressFill.style.width = "0%";
        addMessage("assistant", `❌ Upload failed: ${err.message}`);
        setTimeout(() => uploadProgress.classList.add("hidden"), 3000);
    }

    fileInput.value = "";
}

// ============ Document List ============

function renderDocuments() {
    const ids = Object.keys(documents);
    docCount.textContent = `${ids.length} uploaded`;
    documentList.innerHTML = "";

    ids.forEach((docId) => {
        const fields = documents[docId];
        const card = document.createElement("div");
        card.className = "doc-card";
        card.innerHTML = `
            <div class="doc-card-header">
                <strong>${fields.employee_name || docId}</strong>
                <span class="doc-type-badge">${fields.document_type || "doc"}</span>
            </div>
            <div class="doc-card-fields">
                ${fields.period ? `<div class="doc-field"><span>Period</span><span class="doc-field-value">${fields.period}</span></div>` : ""}
                ${fields.basic_pay != null ? `<div class="doc-field"><span>Basic</span><span class="doc-field-value">₹${Number(fields.basic_pay).toLocaleString("en-IN")}</span></div>` : ""}
                ${fields.gross_salary != null ? `<div class="doc-field"><span>Gross</span><span class="doc-field-value">₹${Number(fields.gross_salary).toLocaleString("en-IN")}</span></div>` : ""}
                ${fields.net_pay != null ? `<div class="doc-field"><span>Net Pay</span><span class="doc-field-value">₹${Number(fields.net_pay).toLocaleString("en-IN")}</span></div>` : ""}
            </div>
        `;
        documentList.appendChild(card);
    });

    // Show compare section if 2+ documents
    if (ids.length >= 2) {
        compareSection.classList.remove("hidden");
        compareDocA.innerHTML = ids.map((id) => `<option value="${id}">${documents[id].employee_name || id} (${documents[id].period || ""})</option>`).join("");
        compareDocB.innerHTML = ids.map((id) => `<option value="${id}">${documents[id].employee_name || id} (${documents[id].period || ""})</option>`).join("");
        if (ids.length >= 2) compareDocB.selectedIndex = 1;
    } else {
        compareSection.classList.add("hidden");
    }
}

// ============ Chat ============

chatInput.addEventListener("input", () => {
    sendBtn.disabled = !chatInput.value.trim();
    // Auto-resize textarea
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (chatInput.value.trim()) sendMessage();
    }
});

sendBtn.addEventListener("click", sendMessage);

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || !sessionId) return;

    addMessage("user", message);
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;

    // Show typing indicator
    const typingId = addTypingIndicator();

    try {
        const resp = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, message }),
        });

        removeTypingIndicator(typingId);

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Chat failed");
        }

        const data = await resp.json();
        addMessage("assistant", data.response);
    } catch (err) {
        removeTypingIndicator(typingId);
        addMessage("assistant", `⚠️ Error: ${err.message}`);
    }
}

function addMessage(role, content) {
    const div = document.createElement("div");
    div.className = `message ${role}-message`;
    const avatar = role === "user" ? "👤" : "🤖";
    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content"><p>${formatMessage(content)}</p></div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMessage(text) {
    // Basic markdown-like formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
}

function addTypingIndicator() {
    const id = "typing-" + Date.now();
    const div = document.createElement("div");
    div.id = id;
    div.className = "message assistant-message";
    div.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content"><p class="loading-spinner">Thinking</p></div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ============ Compare ============

compareBtn.addEventListener("click", async () => {
    const docA = compareDocA.value;
    const docB = compareDocB.value;

    if (docA === docB) {
        addMessage("assistant", "⚠️ Please select two different documents to compare.");
        return;
    }

    addMessage("user", `Compare ${documents[docA]?.period || docA} vs ${documents[docB]?.period || docB}`);
    const typingId = addTypingIndicator();

    try {
        const resp = await fetch(`${API_BASE}/api/compare`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: sessionId,
                doc_id_a: docA,
                doc_id_b: docB,
            }),
        });

        removeTypingIndicator(typingId);

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Comparison failed");
        }

        const data = await resp.json();
        addMessage("assistant", data.explanation);
    } catch (err) {
        removeTypingIndicator(typingId);
        addMessage("assistant", `⚠️ Comparison error: ${err.message}`);
    }
});

// ============ Initialize ============

initSession();
```

- [ ] **Step 4: Verify frontend files are created and well-formed**

Run: `dir frontend\`
Expected: `index.html`, `style.css`, `app.js` are present

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: add premium dark-mode frontend with upload, chat, and compare UI"
```

---

## Task 13: README & Final Integration Test

**Files:**
- Create: `paysense-ai/README.md`

**Interfaces:**
- Consumes: Everything from Tasks 1-12
- Produces: Complete documentation and verified integration

- [ ] **Step 1: Create `README.md`**

```markdown
# PaySense AI 💰

> Smart Payslip & Offer Letter Assistant powered by RAG

Upload your Indian payslip or offer letter (PDF/image) and ask questions in plain English. Compare two documents side-by-side to understand salary changes.

## Features

- 📄 **Upload & Parse** — PDF and image (OCR) support for payslips and offer letters
- 💬 **Ask Anything** — "What is my in-hand salary?", "Explain my HRA", "How much PF am I contributing?"
- 📊 **Compare Documents** — Side-by-side comparison with change explanations
- 🧠 **RAG-Powered** — Pre-loaded Indian payroll knowledge base + your uploaded documents
- 👥 **Multi-User** — Session-based isolation, no data shared between users
- 💸 **Free** — Uses Gemini Flash free tier + local embeddings + local vector DB

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.0 Flash (free tier) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local) |
| Vector DB | ChromaDB (local) |
| Backend | Python, FastAPI |
| PDF Parsing | PyMuPDF |
| OCR | Tesseract |
| Frontend | HTML, CSS, JavaScript |

## Setup

### Prerequisites
- Python 3.11+
- Tesseract OCR ([install guide](https://github.com/tesseract-ocr/tesseract))
- Google Gemini API key ([get one free](https://aistudio.google.com/app/apikey))

### Install

```bash
cd paysense-ai
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### Configure

```bash
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Run

```bash
python run.py
```

Open http://localhost:8000 in your browser.

### Test

```bash
python -m pytest tests/ -v
```

## Project Structure

```
paysense-ai/
├── src/                    # Application source code
│   ├── parsers/            # PDF and image text extraction
│   ├── rag/                # Embeddings, vector store, knowledge base, retrieval
│   ├── agent/              # LangChain agent with RAG tools
│   ├── comparison/         # Document comparison logic
│   ├── sessions/           # In-memory session management
│   └── api/                # FastAPI routes and app factory
├── knowledge_base/         # Pre-loaded Indian payroll knowledge (9 topics)
├── frontend/               # Web UI (HTML/CSS/JS)
├── tests/                  # Comprehensive test suite
├── run.py                  # Entry point
└── requirements.txt        # Python dependencies
```

## How It Works

1. User uploads a payslip/offer letter (PDF or image)
2. Text is extracted (PyMuPDF for PDF, Tesseract for images)
3. Gemini Flash extracts structured fields (basic pay, HRA, PF, etc.)
4. Document chunks are embedded and stored in ChromaDB
5. User asks a question → RAG retrieves relevant context from KB + user docs
6. Gemini Flash generates a grounded, cited answer
7. For comparisons, fields are diffed and changes explained in plain language
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (approximately 30+ tests)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with setup and architecture guide"
```

---

## Verification Plan

### Automated Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run with coverage (optional)
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Expected: All tests pass. Key areas covered:
- Config loading
- PDF parsing
- OCR parsing
- Field extraction (mocked LLM)
- Embeddings and vector store
- Knowledge base loading and retrieval
- Combined retriever
- Document comparator
- Session manager
- Agent chat flow (mocked LLM)
- API endpoints (mocked LLM + components)

### Manual Verification

1. Start the server: `python run.py`
2. Open http://localhost:8000
3. Upload a sample payslip PDF
4. Ask: "What is my in-hand salary?"
5. Upload a second payslip
6. Use the Compare feature
7. Verify responses cite sources and explain terms correctly
