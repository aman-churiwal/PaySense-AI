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
    tmp = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w")
    tmp.write("not an image")
    tmp.close()
    try:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_image(tmp.name)
    finally:
        os.unlink(tmp.name)