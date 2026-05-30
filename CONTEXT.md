# Wilderness Med Chat — Project Context

## Vision

A voice-driven wilderness medicine assistant designed to run fully offline on an edge computing device. The primary user is someone in the backcountry — injured, stressed, potentially alone — who needs fast, reliable, evidence-based first aid and wilderness medicine guidance without any internet connection.

- **Development platform**: Apple M1 Pro MacBook
- **Deployment target**: NVIDIA Jetson Nano (4GB, ARM64 + 128-core Maxwell GPU, CUDA)

---

## Core Architecture

### 1. Language Model — Ollama
- Run a local LLM via [Ollama](https://ollama.com/) (no internet required at inference time)
- Model TBD — likely a quantized 7B or 8B model (e.g. `llama3`, `mistral`, `phi3`) chosen for speed/accuracy tradeoff on edge hardware
- Ollama exposes a local REST API (`http://localhost:11434`)

### 2. Knowledge Base — RAG with LlamaIndex + ChromaDB
- **Retrieval-Augmented Generation (RAG)** to ground the LLM in authoritative wilderness medicine literature
- **LlamaIndex** as the RAG orchestration framework (document ingestion, chunking, querying)
- **ChromaDB** as the local vector store (persistent, no server required)
- Source documents: PDFs of wilderness medicine textbooks, WEMS/WAFA/WEMT protocols, NOLS Wilderness Medicine guides, etc.
- Ingestion pipeline: PDF → text chunks → embeddings → ChromaDB index

### 3. Speech-to-Text — faster-whisper
- **faster-whisper** (CTranslate2-based Whisper implementation)
- Chosen over `mlx-whisper` because MLX is Apple Silicon only — faster-whisper runs on:
  - M1 Mac (CPU, fast enough for real-time)
  - Jetson Nano (CUDA — same code, no changes at deploy time)
- Records audio from microphone, transcribes locally with no network call
- Model: `base.en` or `small.en` (tradeoff between speed and accuracy)

### 4. Text-to-Speech — Kokoro-82M (ONNX)
- **Kokoro-82M** via ONNX runtime
- Chosen because: natural sounding, small (82M params), ONNX means it runs on both Mac (CPU/CoreML) and Jetson Nano (CUDA or CPU) without code changes
- Fully offline, no server required
- macOS `say` command kept as a zero-dependency dev fallback

---

## Data Flow (End-to-End)

```
[User speaks]
     ↓
[MLX Whisper — STT]
     ↓
[Transcribed text query]
     ↓
[LlamaIndex RAG query → ChromaDB vector search]
     ↓
[Relevant document chunks retrieved]
     ↓
[Prompt assembled: system prompt + context chunks + user query]
     ↓
[Ollama LLM — local inference]
     ↓
[LLM response text]
     ↓
[TTS engine — speech output]
     ↓
[User hears answer]
```

---

## Development Environment

- **Platform**: Apple M1 Pro MacBook (development); edge device (eventual deployment target)
- **Language**: Python 3.11+
- **Package manager**: `uv` (dev), `pip` (deployment)
- **Ollama**: running locally via `ollama serve`
- **ChromaDB**: local persistent store, no Docker needed
- **LLM model**: `gemma3:4b` — fits in Jetson Nano 4GB RAM with 4-bit quantization; good reasoning for its size

---

## Key Constraints & Goals

- **Fully offline** — zero network calls at runtime (internet only needed during setup/ingestion)
- **Low latency** — response should feel conversational, not slow
- **Accuracy / safety** — RAG must pull from vetted wilderness medicine sources; hallucination risk must be minimized
- **Edge-deployable** — architecture should map cleanly to ARM Linux (Raspberry Pi 5, NVIDIA Jetson, etc.)
- **Voice-first UX** — hands-free operation is the primary interface; text fallback acceptable

---

## Planned Source Documents (RAG Corpus)

- NOLS Wilderness Medicine (textbook)
- Wilderness Medical Associates (WMA) protocols
- Wilderness Medicine Institute (WMI) field guides
- WEMS/WAFA/WEMT curriculum materials
- Additional PDFs TBD — all stored in `data/pdfs/`

---

## Project Structure (Planned)

```
wilderness_med_chat/
├── CONTEXT.md              # This file
├── data/
│   └── pdfs/               # Source wilderness medicine PDFs
├── ingest/
│   └── ingest.py           # PDF ingestion pipeline → ChromaDB
├── rag/
│   └── query.py            # LlamaIndex RAG query interface
├── stt/
│   └── transcribe.py       # MLX Whisper speech-to-text
├── tts/
│   └── speak.py            # TTS engine wrapper
├── llm/
│   └── ollama_client.py    # Ollama API client
├── main.py                 # Main voice loop
└── requirements.txt        # Python dependencies
```

---

## Build Phases

### Phase 1 — Core (text in / text out) ✅ DONE
- Ollama + gemma3:4b working
- PDF ingestion → ChromaDB (`ingest/ingest.py`)
- RAG query → LLM response (`rag/query.py`, `llm/ollama_client.py`)
- `main_text.py` CLI loop — streaming responses, sliding conversation window

### Phase 2 — Voice out
- Kokoro-82M ONNX TTS wired in
- `main_tts.py`

### Phase 3 — Full voice loop
- faster-whisper STT wired in
- `main.py` — mic → STT → RAG → LLM → TTS → speaker

### Phase 4 — Edge deployment
- Test on Jetson Nano
- Optimize model quantization / chunk sizes
- Startup script, systemd service

---

## Open Questions / Decisions Pending

- [ ] Embedding model for ChromaDB — `nomic-embed-text` via Ollama (keeps everything in Ollama) vs `sentence-transformers/all-MiniLM-L6-v2` (faster, self-contained)
- [ ] Chunk size / overlap strategy for medical PDFs (medical content is dense; likely 512 tokens, 64 overlap)
- [ ] Multi-turn conversation context — sliding window of last N turns prepended to prompt
- [ ] Wake word vs push-to-talk activation
- [ ] UI — voice only, or small LCD/e-ink display on Nano?
- [ ] Jetson Nano JetPack version and CUDA compatibility for Ollama
