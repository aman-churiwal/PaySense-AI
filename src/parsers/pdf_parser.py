"""Extract text content from PDF files using PyMuPDF."""

import os
import fitz  # PyMuPDF


def parse_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path (str): Absolute or relative path to the PDF file.

    Returns:
        Extracted text as a single string, pages separated by newlines.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If the file cannot be parsed as a PDF.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Could not parse file as PDF: {file_path}. Error: {e}")

    if not doc.is_pdf:
        doc.close()
        raise ValueError(f"Could not parse file as PDF: {file_path}. Not a valid PDF.")

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
        raise ValueError(f"Could not parse file as PDF: {file_path}. No extractable text found.")

    return result