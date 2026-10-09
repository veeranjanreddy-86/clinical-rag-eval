FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data
RUN python -m clinical_rag.cli ingest && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "clinical_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
