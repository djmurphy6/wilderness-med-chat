# Wilderness Med Chat — Project Context

## Vision

A voice-driven wilderness medicine assistant designed to run fully offline on an edge computing device. The primary user is someone in the backcountry — injured, stressed, potentially alone — who needs fast, reliable, evidence-based first aid and wilderness medicine guidance without any internet connection.

- **Development platform**: Apple M1 Pro MacBook
- **Deployment target**: NVIDIA Jetson Nano or Raspberry Pi (4GB class edge devices, ARM64)

---

## Core Architecture

### 1. Language Model — Ollama + gemma3:4b
- Local LLM via [Ollama](https://ollama.com/) — no internet at inference time
- Model: **`gemma3:4b`** — fits in Jetson Nano 4GB RAM with 4-bit quantization; good reasoning for its size
- Ollama REST API at `http://localhost:11434`
- System prompt encodes the full **Patient Assessment System (PAS)** — model walks the rescuer through scene size-up → primary survey → SAMPLE history → vitals → focused exam → problem list → monitoring
- Each turn the system prompt is augmented with a live **Patient State** summary (see below)

### 2. Structured Patient Memory — PatientState
- **`patient/state.py`** — `PatientState` dataclass updated each turn from user input
- **Deterministic extraction** (no LLM) — keyword matching, not inference — fast and reliable
- **Tracked fields**: mechanism of injury, nature of illness, mental status (AVPU), airway, breathing, major bleeding, spine concern, chief complaint, current PAS step, active problem list
- **`to_prompt_section()`** — renders a compact case summary injected into the system prompt every turn, keeping the model anchored to the actual patient
- **`to_retrieval_query()`** — enriches the ChromaDB query with accumulated case facts so short follow-up turns (e.g. "yes still breathing") don't pull irrelevant chunks
- Resets to a fresh `PatientState()` when the user types `new` / `reset`

### 3. Knowledge Base — RAG with LlamaIndex + ChromaDB
- **Retrieval-Augmented Generation (RAG)** grounds the LLM in authoritative wilderness medicine literature
- **LlamaIndex** for document ingestion, chunking, and querying
- **ChromaDB** as the local persistent vector store (no server required)
- **Embedding model**: `nomic-embed-text` via Ollama
- **Chunk settings**: 512 tokens, 64-token overlap
- **Corpus ingested** (2,487 chunks across 3 PDFs):
  - Aerie Backcountry Medicine, 15th ed.
  - Wilderness Medicine: Beyond First Aid, 7th ed.
  - WFA Text
- Ingestion pipeline: PDF → text chunks → `nomic-embed-text` embeddings → ChromaDB

### 4. Speech-to-Text — faster-whisper
- **faster-whisper** (CTranslate2-based Whisper implementation)
- Chosen over `mlx-whisper` because MLX is Apple Silicon only — faster-whisper runs on:
  - M1 Mac (CPU, fast enough for real-time)
  - Jetson Nano (CUDA — same code, no changes at deploy time)
- Records audio from microphone, transcribes locally with no network call
- Model: `base.en` or `small.en` (tradeoff between speed and accuracy)

### 5. Text-to-Speech — Kokoro-82M (ONNX)
- **Kokoro-82M** via ONNX runtime
- Natural sounding, small (82M params), cross-platform (Mac CPU + Jetson CUDA) via ONNX
- Fully offline, no server required
- macOS `say` command kept as a zero-dependency dev fallback

---

## Data Flow (End-to-End)

```
[User speaks]
     ↓
[faster-whisper — STT]
     ↓
[Transcribed text query]
     ↓
[LlamaIndex RAG query → ChromaDB vector search]
     ↓
[Relevant document chunks retrieved]
     ↓
[Prompt assembled: PAS system prompt + context chunks + conversation history + user query]
     ↓
[Ollama LLM (gemma3:4b) — local inference]
     ↓
[LLM response text]
     ↓
[Kokoro-82M ONNX — TTS]
     ↓
[User hears answer]
```

---

## Development Environment

- **Platform**: Apple M1 Pro MacBook (development); Jetson Nano (deployment)
- **Language**: Python 3.11
- **Virtual env**: `.venv/` (standard venv + pip)
- **Ollama**: running locally via `ollama serve`
- **ChromaDB**: persistent local store at `data/chroma/`
- **Models pulled**: `gemma3:4b` (3.3GB), `nomic-embed-text` (274MB)

---

## Key Constraints & Goals

- **Fully offline** — zero network calls at runtime (internet only needed during setup/ingestion)
- **Low latency** — response should feel conversational, not slow
- **Safety** — model is an aid, not a decision-maker; no definitive diagnoses; always route to evacuate when uncertain
- **PAS-driven** — model follows the Patient Assessment System, asks 2–3 questions at a time, never dumps the full protocol
- **Edge-deployable** — architecture maps cleanly to ARM64 Linux + CUDA (Jetson Nano)
- **Voice-first UX** — hands-free operation is the primary interface; text CLI available for dev/testing

---

## RAG Corpus

All PDFs stored in `data/pdfs/`. ChromaDB index at `data/chroma/`.

| File | Description |
|---|---|
| `Aerie-15th-Edition-Manual-15-02.pdf` | Aerie Backcountry Medicine, 15th ed. |
| `wilderness-medicine-beyond-first-aid-7th-edition-*.pdf` | Wilderness Medicine: Beyond First Aid, 7th ed. |
| `WFA+Text.pdf` | WFA curriculum text |

**Current index**: 2,487 chunks (512 tokens, 64 overlap)

---

## Project Structure

```
wilderness_med_chat/
├── README.md
├── CONTEXT.md
├── .env                        # GOOGLE_API_KEY (eval only, gitignored)
├── .env.example
├── requirements.txt
├── pytest.ini
├── Makefile
├── main_text.py                # Text-only CLI loop
├── main.py                     # Voice-first CLI loop (Enter -> mic STT -> LLM -> TTS)
├── data/
│   ├── pdfs/                   # Source wilderness medicine PDFs
│   └── chroma/                 # ChromaDB vector index (2,487 chunks)
├── llm/
│   └── ollama_client.py        # Ollama wrapper, PAS system prompt, message assembly
├── patient/
│   └── state.py                # PatientState — deterministic structured memory across turns
├── ingest/
│   └── ingest.py               # PDF → chunks → embeddings → ChromaDB
├── rag/
│   └── query.py                # ChromaDB retrieval, lazy-loaded singleton
├── tts/
│   └── speak.py                # TTS backend selection (macOS say now, Kokoro path scaffolded)
├── stt/
│   └── transcribe.py           # faster-whisper STT helpers (record + transcribe)
└── tests/
    ├── conftest.py              # Shared fixtures, ollama_available() guard
    ├── unit/
    │   ├── test_ollama_client.py    # 12 tests — message building, prompt structure
    │   └── test_patient_state.py    # 7 tests — deterministic extraction, retrieval enrichment
    ├── integration/
    │   ├── test_llm_live.py         # 4 tests — live Ollama
    │   └── test_rag_pipeline.py     # 5 tests — live RAG
    ├── scenarios/
    │   ├── fixtures/scenarios.yaml  # 5 patient cases (ankle, head, anaphylaxis, hypothermia, chest)
    │   └── test_pas_behavior.py     # PAS behavioral assertions + multi-turn checks
    └── eval/
        ├── generate_dataset.py      # Generates eval_dataset.jsonl via Gemini 2.5 Flash
        └── test_ragas_eval.py       # RAGAS faithfulness + context precision (cloud judge)
```

---

## Build Phases

### Phase 1 — Core (text in / text out) ✅ DONE
- `gemma3:4b` + `nomic-embed-text` pulled via Ollama
- 3 PDFs ingested → 2,487 ChromaDB chunks
- PAS system prompt — structured 7-step patient assessment workflow
- **Structured patient memory** — `PatientState` tracks MOI, AVPU, airway, breathing, bleeding, spine concern, problem list across turns; enriches both the system prompt and RAG retrieval query
- `main_text.py` — streaming CLI loop, 10-turn history window, `new`/`reset` command
- Full test suite: 12 unit (ollama_client) + 7 unit (patient_state) + integration + scenario + RAGAS eval infrastructure

### Phase 2 — Voice out ✅ DONE
- Runtime speaks assistant responses via local TTS backend (`tts/speak.py`)
- Current default backend: macOS `say`; Kokoro runtime wiring remains a follow-up

### Phase 3 — Full voice loop ✅ DONE
- faster-whisper STT (`stt/transcribe.py`) for short microphone capture + transcription
- `main.py` — voice-first loop with typed fallback, RAG retrieval, LLM streaming, and TTS playback

### Phase 4 — Edge deployment ⬜
- Test on Jetson Nano
- Optimize chunk sizes (RAGAS context precision score will guide this)
- Startup script, systemd service

---

## Test Commands

```bash
make test-unit          # 12 fast tests, no Ollama needed
make test-integration   # live RAG + LLM tests
make test-scenarios     # PAS behavioral evals (5 patient scenarios)
make eval-generate      # generate eval_dataset.jsonl via Gemini (needs GOOGLE_API_KEY)
make eval               # RAGAS faithfulness + context precision
make ingest             # re-ingest PDFs into ChromaDB
make run                # launch voice-first loop
make run-text           # launch text-only loop
```

---

## Open Questions / Decisions Pending

- [ ] Wake word vs push-to-talk activation
- [ ] UI — voice only, or small LCD/e-ink display on Nano?
- [ ] Jetson Nano JetPack version and CUDA compatibility for Ollama
- [ ] Chunk size tuning — run RAGAS context precision after eval dataset is generated to validate 512/64 or try 256/32
- [ ] Multi-turn context window size — currently 10 turns (20 messages); may need adjustment for longer assessments
