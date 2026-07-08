"""Tests for document comparison logic."""

import pytest
from src.comparison.comparator import compare_documents


DOC_A = {
    "document_type": "payslip",
    "employee_name": "Rahul Sharma",
    "company_name": "TechCorp",
    "period": "January 2025",
    "basic_pay": 25000,
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
    "basic_pay": 28000,
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
    assert "basic_pay" in changed_fields
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