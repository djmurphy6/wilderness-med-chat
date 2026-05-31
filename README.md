# Wilderness Med Chat

An offline, voice-first wilderness medicine assistant for the backcountry. It guides rescuers through the **Patient Assessment System (PAS)** using RAG-grounded answers from authoritative wilderness medicine texts — with zero internet required at runtime.

Built for edge deployment on **NVIDIA Jetson Nano**; developed on Apple Silicon Mac.

> **Disclaimer:** This is an educational aid, not a substitute for professional medical care or certified wilderness medicine training. When in doubt, evacuate.

---

## Features

- **Fully offline inference** — local LLM, embeddings, STT, and TTS; no cloud calls during use
- **RAG over wilderness medicine literature** — answers grounded in ingested PDFs, not model memory alone
- **PAS-driven workflow** — structured 7-step patient assessment (scene size-up → primary survey → SAMPLE → vitals → focused exam → problem list → monitoring)
- **Structured patient memory** — `PatientState` tracks MOI, AVPU, airway, breathing, bleeding, and more across turns
- **Voice-first design** — hands-free operation for stressed rescuers (text CLI available today for dev/testing)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| LLM | [Ollama](https://ollama.com/) + **Gemma 3 4B** |
| RAG | [LlamaIndex](https://www.llamaindex.ai/) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local, persistent) |
| Embeddings | `nomic-embed-text` via Ollama |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper / CTranslate2) |
| Text-to-speech | [Kokoro-82M](https://github.com/hexgrad/kokoro) via ONNX Runtime |
| Testing | pytest, RAGAS |
| Eval (dev only) | Google Gemini 2.5 Flash |

---

## Architecture

```
[User speaks]
     ↓
[faster-whisper — STT]
     ↓
[Transcribed text query]
     ↓
[LlamaIndex RAG → ChromaDB vector search]
     ↓
[Relevant document chunks retrieved]
     ↓
[Prompt: PAS system prompt + PatientState + context + history + query]
     ↓
[Ollama LLM (gemma3:4b)]
     ↓
[Kokoro-82M ONNX — TTS]
     ↓
[User hears answer]
```

**Knowledge base:** 3 wilderness medicine PDFs → 2,487 chunks (512 tokens, 64 overlap) stored in `data/chroma/`.

---

## Quick Start

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.com/) installed and running

### Setup

```bash
# Clone and enter the project
cd wilderness_med_chat

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull gemma3:4b
ollama pull nomic-embed-text

# Start Ollama (if not already running)
ollama serve
```

Place wilderness medicine PDFs in `data/pdfs/`, then ingest:

```bash
make ingest
```

### Run

```bash
make run
```

Type wilderness medicine questions at the prompt. Commands:

- `new` or `reset` — start a fresh patient assessment
- `exit` — quit

---

## Development

### Project Structure

```
wilderness_med_chat/
├── main_text.py          # Text CLI (working)
├── llm/                  # Ollama client, PAS system prompt
├── patient/              # PatientState — structured memory across turns
├── rag/                  # ChromaDB retrieval
├── ingest/               # PDF → chunks → embeddings → ChromaDB
├── stt/                  # faster-whisper STT (planned)
├── tts/                  # Kokoro ONNX TTS (planned)
├── data/
│   ├── pdfs/             # Source PDFs
│   └── chroma/           # Vector index
└── tests/
    ├── unit/
    ├── integration/
    ├── scenarios/
    └── eval/
```

### Make Commands

```bash
make run                # Launch text chat loop
make ingest             # Re-ingest PDFs into ChromaDB
make test-unit          # Fast unit tests (no Ollama needed)
make test-integration   # Live RAG + LLM tests
make test-scenarios     # PAS behavioral evals (5 patient scenarios)
make test               # All local tests
make eval-generate      # Generate eval dataset (needs GOOGLE_API_KEY)
make eval               # RAGAS faithfulness + context precision
```

### Eval Setup (optional)

Eval uses a cloud judge for quality metrics only — not used in production.

```bash
cp .env.example .env
# Add GOOGLE_API_KEY to .env

make eval-generate
make eval
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Core | ✅ Done | Text CLI, RAG, PAS prompts, PatientState, test suite |
| 2 — Voice out | 🔜 Next | Kokoro TTS — text in, spoken response out |
| 3 — Full voice | ⬜ Planned | faster-whisper STT + full mic → speaker loop |
| 4 — Edge deploy | ⬜ Planned | Jetson Nano deployment, systemd service |

---

## Hardware

- **Development:** Apple M1 Pro MacBook
- **Deployment target:** NVIDIA Jetson Nano (4GB, ARM64 + CUDA)

All chosen models and libraries (Ollama, faster-whisper, Kokoro ONNX) run on both Mac CPU and Jetson CUDA without code changes.

---

## License

See repository for license details. Wilderness medicine PDFs in `data/pdfs/` are not included — add your own licensed copies for ingestion.
