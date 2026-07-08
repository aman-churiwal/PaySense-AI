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
    "basic_pay": 25000,
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
    assert result["basic_pay"] == 25000
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
    assert "basic_pay" in EXTRACTION_PROMPT
    assert "hra" in EXTRACTION_PROMPT
    assert "net_pay" in EXTRACTION_PROMPT
    assert "document_type" in EXTRACTION_PROMPT