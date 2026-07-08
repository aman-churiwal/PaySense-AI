"""Tests for PDF text extraction."""

import os
import tempfile
import fitz  # PyMuPDF - used to create test PDFs
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
    tmp.close()
    doc.save(tmp.name)
    doc.close()
    yield tmp.name
    os.unlink(tmp.name)


def test_parse_pdf_extracts_text(sample_pdf_path):
    """Should return a string."""
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
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    tmp.write("not a pdf")
    tmp.close()
    try:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_pdf(tmp.name)
    finally:
        os.unlink(tmp.name)

def test_parse_pdf_missing_file():
    """Should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        parse_pdf("/nonexistet/path/file.pdf")