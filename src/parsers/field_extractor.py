"""Extract structured payslip/offer letter fields using Gemini LLM."""

import json
from google import genai

EXTRACTION_PROMPT = """You are a payroll document parser. Extract structured fields from the following document text.

Return ONLY a valid JSON object with these exact keys:
{
  "document_type": "payslip" or "offer_letter" or "unknown",
  "employee_name": string or null,
  "company_name": string or null,
  "period": string or null (e.g. "June 2026"),
  "basic_pay": number or null,
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
- Do NOT invent or calculate values - only extract what is explicitly stated
- Return ONLY the JSON object, no explanation or markdown

Document text:
"""

_FALLBACK_FIELDS = {
    "document_type": "unknown",
    "employee_name": None,
    "company_name": None,
    "period": None,
    "basic_pay": None,
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
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)

    return response.text

def extract_fields(raw_text: str, api_key: str, model_name: str) -> dict:
    """
    Extract structured fields from raw payslip/offer letter text using Gemini.

    Args:
        raw_text: The full text content of the document.
        api_key: Google Gemini API key
        model_name: Gemini model name (e.g. "gemini-2.5-flash").

    Returns:
        ValueError: If raw_text is empty.
    """

    if not raw_text.strip():
        raise ValueError("Cannot extract fields from empty text.")

    prompt = EXTRACTION_PROMPT + raw_text
    response_text = _call_gemini(prompt, api_key, model_name)

    # Strip Markdown code fences if present
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