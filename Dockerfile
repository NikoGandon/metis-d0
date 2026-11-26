FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    METIS_CASSETTE=replay \
    METIS_CASSETTE_DIR=/app/tests/fixtures/cassettes

WORKDIR /app

RUN useradd --create-home --shell /bin/bash metis

COPY pyproject.toml README.md ./
COPY metis ./metis
COPY tests/fixtures ./tests/fixtures
COPY evals/gold_d0.jsonl evals/rapport_2026-11-04_opus.json evals/rapport_2026-11-04_qwen.json ./evals/

RUN pip install --no-cache-dir -e ".[api]" && chown -R metis:metis /app

USER metis

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "metis.api:app", "--host", "0.0.0.0", "--port", "8000"]
