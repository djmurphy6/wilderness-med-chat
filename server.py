"""
FastAPI web server for the Wilderness Med Chat UI.

Wraps the existing RAG + LLM + PatientState pipeline and exposes:
  GET  /              - serves the chat UI
  GET  /health        - Ollama + RAG status
  GET  /patient-state - current assessment snapshot
  POST /chat          - streaming SSE chat (token-by-token)
  POST /transcribe    - raw 16 kHz float32 PCM → transcription
  POST /reset         - clear session and patient state

Run with:
    python server.py          (development)
    make serve                (via Makefile)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))

from llm.ollama_client import build_messages, chat, is_ollama_running
from patient.state import PatientState
from rag.query import rag_engine

MAX_HISTORY_MESSAGES = 20
BASE_DIR = Path(__file__).parent

app = FastAPI(title="Wilderness Med Chat")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Single-user in-memory session (local app)
_session: dict = {
    "history": [],
    "patient_state": PatientState(),
}


def _state_to_dict(ps: PatientState) -> dict:
    return {
        "mechanism": ps.mechanism,
        "nature_of_illness": ps.nature_of_illness,
        "mental_status": ps.mental_status,
        "airway": ps.airway,
        "breathing": ps.breathing,
        "major_bleeding": ps.major_bleeding,
        "spine_concern": ps.spine_concern,
        "chief_complaint": ps.chief_complaint,
        "current_pas_step": ps.current_pas_step.replace("_", " "),
        "active_problem_list": ps.active_problem_list,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((BASE_DIR / "static" / "index.html").read_text())


@app.get("/health")
async def health():
    ollama_ok = is_ollama_running()
    rag_ok = not rag_engine.is_empty()
    return {"ollama": ollama_ok, "rag": rag_ok, "ready": ollama_ok}


@app.get("/patient-state")
async def get_patient_state():
    return _state_to_dict(_session["patient_state"])


@app.post("/reset")
async def reset_session():
    _session["history"] = []
    _session["patient_state"] = PatientState()
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_input = body.get("message", "").strip()
    if not user_input:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    ps = _session["patient_state"]
    history = _session["history"]

    # Run the same pipeline as main.py: state update → RAG → LLM
    ps.update_from_text(user_input)
    retrieval_query = ps.to_retrieval_query(user_input)
    context_chunks = rag_engine.retrieve(retrieval_query)
    messages = build_messages(user_input, context_chunks, history, ps)

    def generate():
        full_response = ""
        try:
            for chunk in chat(messages, stream=True):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        except Exception as err:
            yield f"data: {json.dumps({'type': 'error', 'message': str(err)})}\n\n"
            return

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})
        if len(history) > MAX_HISTORY_MESSAGES:
            _session["history"] = history[-MAX_HISTORY_MESSAGES:]

        yield f"data: {json.dumps({'type': 'done', 'patient_state': _state_to_dict(ps)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/transcribe")
async def transcribe_audio(request: Request):
    """
    Accepts raw mono float32 PCM at 16 kHz (little-endian, application/octet-stream)
    resampled by the browser's OfflineAudioContext, returns JSON {"text": "..."}.
    """
    try:
        from stt.transcribe import WhisperTranscriber

        body = await request.body()
        if not body:
            return JSONResponse({"error": "No audio data"}, status_code=400)

        audio = np.frombuffer(body, dtype="<f4")  # little-endian float32

        # Minimum ~0.1 s of audio
        if audio.size < 1600:
            return {"text": ""}

        if not hasattr(app.state, "transcriber"):
            app.state.transcriber = WhisperTranscriber()

        text = app.state.transcriber.transcribe_audio(audio)
        return {"text": text}

    except ImportError:
        return JSONResponse(
            {"error": "faster-whisper not installed; voice input unavailable"},
            status_code=503,
        )
    except Exception as err:
        return JSONResponse({"error": str(err)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
