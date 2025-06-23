# ---------------- Base image ----------------
FROM python:3.11-slim

# ---------------- Environment ----------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------------- OS deps --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev curl libxml2-dev libxslt1-dev \
        antiword poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ---------------- Workdir --------------------
WORKDIR /app                 # <-- IMPORTANT

# ---------------- Python deps ----------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# spaCy English model
RUN python -m spacy download en_core_web_sm

# ---------------- App code -------------------
COPY . .                      # copies everything in backend/ into /app

# ---------------- Runtime dirs & ownership ----
RUN mkdir -p /app/backups /app/uploads /app/chroma_db /app/logs \
    && groupadd -r tailingsiq && useradd -r -g tailingsiq tailingsiq \
    && chown -R tailingsiq:tailingsiq /app

USER tailingsiq

# ---------------- Healthcheck & Port ---------
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# ---------------- Start ----------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
