"""Extract text from images using Tesseract OCR."""

import os
import pytesseract
from PIL import Image


def parse_image(file_path: str) -> str:
    """Extract text from an image file using Tesseract OCR.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns: Extracted text as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as an image.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}.")

    try:
        img = Image.open(file_path)
    except Exception as e:
        raise ValueError(f"Could not parse file as image: {file_path}. Error: {e}")

    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        raise ValueError(f"Could not parse file as image: {file_path}. OCR error: {e}")

    if not text.strip():
        raise ValueError(f"Could not parse file as image: {file_path}. No text detected.")

    return text.strip()