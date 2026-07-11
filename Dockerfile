FROM python:3.12-slim

# tesseract-ocr: required by pytesseract for image (PNG/JPG) OCR
# libgl1: required by some PyMuPDF/Pillow image-handling paths
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Chroma's persist directory needs to be writable regardless of which
# user the container ends up running as
RUN mkdir -p /app/chroma_data && chmod -R 777 /app/chroma_data

# Hugging Face Spaces (Docker SDK) expects the app on port 7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn run:app --host 0.0.0.0 --port ${ACTOR_WEB_SERVER_PORT:-7860}"]
