"""Compare two payslip/offer letter documents field-by-field."""

# Fields to compare (in display order) with human-readable labels
COMPARABLE_FIELDS = [
    ("basic_pay", "Basic Pay"),
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
        - summary: A short text summary of the comparison.
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