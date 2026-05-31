VENV = .venv/bin/python
PYTEST = .venv/bin/pytest

.PHONY: test test-unit test-integration test-scenarios eval-generate eval ingest run run-text serve help

## Run only fast unit tests (no Ollama needed)
test-unit:
	$(PYTEST) -m unit

## Run integration tests (requires Ollama + models downloaded)
test-integration:
	$(PYTEST) -m integration

## Run behavioral scenario evals (requires Ollama + gemma3:4b)
test-scenarios:
	$(PYTEST) -m scenario -s

## Run all local tests (unit + integration + scenario)
test:
	$(PYTEST) -m "unit or integration or scenario"

## Generate synthetic eval dataset from ChromaDB chunks via Gemini 2.5 Flash
## Requires: make ingest first, GOOGLE_API_KEY in .env
eval-generate:
	$(VENV) -m tests.eval.generate_dataset

## Run RAGAS faithfulness + context precision eval (cloud judge via Gemini)
## Requires: make eval-generate first, GOOGLE_API_KEY in .env
eval:
	$(PYTEST) -m eval -s -v

## Quick smoke-test eval on 5 random questions (~2 min instead of ~20 min)
eval-quick:
	RAGAS_SAMPLE=5 $(PYTEST) -m eval -s -v

## Ingest all PDFs in data/pdfs/
ingest:
	$(VENV) -m ingest.ingest

## Launch voice-first chat loop (Enter to speak, type for fallback)
run:
	$(VENV) main.py

## Launch text-only chat loop
run-text:
	$(VENV) main_text.py

## Launch the web UI (browser chat with voice + text input)
serve:
	.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8088 --reload

## Show this help
help:
	@grep -E '^##' Makefile | sed 's/## //'
