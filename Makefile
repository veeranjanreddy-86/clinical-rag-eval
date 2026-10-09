PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: install test lint format run ingest eval docker-build docker-run clean

install:  ## Create venv and install package + dev deps
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .

test:  ## Run the test suite
	$(BIN)/pytest

lint:  ## Lint and check formatting
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

ingest:  ## Chunk data/docs into artifacts/chunks.jsonl
	$(BIN)/python -m clinical_rag.cli ingest

eval: ingest  ## Run the gold-set evaluation and write reports/eval_report.md
	$(BIN)/python -m clinical_rag.cli eval

run: ingest  ## Serve the API on http://127.0.0.1:8000
	$(BIN)/uvicorn clinical_rag.api:app --host 127.0.0.1 --port 8000

docker-build:
	docker build -t clinical-rag-eval .

docker-run:
	docker run --rm -p 8000:8000 clinical-rag-eval

clean:
	rm -rf artifacts .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
