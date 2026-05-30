VENV = .venv/bin/python
PYTEST = .venv/bin/pytest

.PHONY: test test-unit test-integration test-scenarios ingest run help

## Run only fast unit tests (no Ollama needed)
test-unit:
	$(PYTEST) -m unit

## Run integration tests (requires Ollama + models downloaded)
test-integration:
	$(PYTEST) -m integration

## Run behavioral scenario evals (requires Ollama + gemma3:4b)
test-scenarios:
	$(PYTEST) -m scenario -s

## Run all tests
test:
	$(PYTEST)

## Ingest all PDFs in data/pdfs/
ingest:
	$(VENV) -m ingest.ingest

## Launch text chat loop
run:
	$(VENV) main_text.py

## Show this help
help:
	@grep -E '^##' Makefile | sed 's/## //'
