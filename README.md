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