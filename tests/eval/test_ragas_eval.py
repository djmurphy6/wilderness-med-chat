"""
RAG evaluation — faithfulness + context precision.

Uses Gemini 2.5-flash as the judge via direct google-genai calls.
The production pipeline (gemma3:4b + ChromaDB) remains fully local.

Faithfulness:
  For each answer, Gemini classifies whether each claim in the answer is
  supported by the retrieved context. Score = supported / total claims.

Context Precision:
  For each question, Gemini rates how relevant the top retrieved chunks are
  to answering the question. Score = relevant chunks / total chunks.

Prerequisites:
  1. PDFs ingested:         make ingest
  2. Eval dataset generated: make eval-generate
  3. GOOGLE_API_KEY set in .env

Run with:
  make eval          # full 40-question run (~20 min)
  make eval-quick    # 5-question smoke test  (~2 min)
"""

import json
import os
import random
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.eval

DATASET_PATH = Path(__file__).parent / "eval_dataset.jsonl"

FAITHFULNESS_THRESHOLD = 0.75
CONTEXT_PRECISION_THRESHOLD = 0.70

# Set RAGAS_SAMPLE=N for a quick smoke test (e.g. RAGAS_SAMPLE=5)
_SAMPLE = int(os.getenv("RAGAS_SAMPLE", "0"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        pytest.skip(
            f"Eval dataset not found at {DATASET_PATH}. "
            "Run `make eval-generate` first."
        )
    with open(DATASET_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if _SAMPLE > 0:
        records = random.sample(records, min(_SAMPLE, len(records)))
        print(f"\n[RAGAS_SAMPLE={_SAMPLE}] Running on {len(records)} questions.")
    return records


def check_google_key():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set in .env — skipping eval.")


def gemini_call(prompt: str, retries: int = 6) -> str:
    """Single Gemini call with exponential backoff on 429 / transient errors."""
    import google.genai as genai
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"thinking_config": {"thinking_budget": 0}},
            )
            return resp.text.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                # parse retry delay hint if present ("retry in Xs")
                import re
                m = re.search(r"retry[^\d]*(\d+)", msg)
                wait = int(m.group(1)) + 2 if m else 60
                print(f"\n  [429] quota hit — waiting {wait}s before retry {attempt+1}/{retries}...", flush=True)
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
    return ""


def run_pipeline(records: list[dict]) -> list[dict]:
    """Run each question through the live RAG + LLM pipeline."""
    from llm.ollama_client import build_messages, chat
    from rag.query import rag_engine

    rows = []
    for i, record in enumerate(records):
        question = record["question"]
        context_chunks = rag_engine.retrieve(question)
        messages = build_messages(question, context_chunks)
        answer = chat(messages, stream=False)
        rows.append({
            "question": question,
            "answer": answer,
            "contexts": context_chunks,
            "ground_truth": record["ground_truth"],
        })
        print(f"  [{i+1}/{len(records)}] retrieved {len(context_chunks)} chunks", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Scorers — exactly 1 Gemini call per question per metric
# ---------------------------------------------------------------------------

def score_faithfulness_one(answer: str, contexts: list[str]) -> float | None:
    """
    Single Gemini call: given the answer and the full retrieved context, rate
    faithfulness 0.0–1.0 (how well the answer is supported by the context).
    Returns None on parse failure.
    """
    context_block = "\n\n---\n\n".join(contexts)
    prompt = (
        "You are an evaluation judge.\n\n"
        "Rate how faithfully the following ANSWER is supported by the CONTEXT. "
        "A score of 1.0 means every claim in the answer is directly supported. "
        "A score of 0.0 means the answer contains significant information not in the context.\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0, nothing else.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"ANSWER:\n{answer}"
    )
    raw = gemini_call(prompt)
    try:
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return None


def score_context_precision_one(question: str, contexts: list[str]) -> float | None:
    """
    Single Gemini call: given the question and all retrieved chunks, rate
    context precision 0.0–1.0 (what fraction of chunks are actually relevant).
    Returns None on parse failure.
    """
    if not contexts:
        return 0.0
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c}" for i, c in enumerate(contexts)
    )
    prompt = (
        "You are an evaluation judge.\n\n"
        "Given the QUESTION and the retrieved CONTEXT CHUNKS below, rate what "
        "fraction of the chunks are relevant to answering the question. "
        "A score of 1.0 means all chunks are on-topic; 0.0 means none are.\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0, nothing else.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT CHUNKS:\n{context_block}"
    )
    raw = gemini_call(prompt)
    try:
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_faithfulness():
    """
    Faithfulness: the generated answer should not introduce claims that go
    beyond the retrieved context. This is the core safety metric.
    """
    check_google_key()
    records = load_dataset()

    print(f"\nRunning pipeline on {len(records)} questions...")
    rows = run_pipeline(records)

    scores = []
    for i, row in enumerate(rows):
        s = score_faithfulness_one(row["answer"], row["contexts"])
        label = f"{s:.2f}" if s is not None else "skip"
        print(f"  [{i+1}/{len(rows)}] faithfulness={label}")
        if s is not None:
            scores.append(s)

    assert scores, "All faithfulness scores were None — Gemini parse failures?"
    score = sum(scores) / len(scores)
    print(f"\nFaithfulness score: {score:.3f}  (threshold: {FAITHFULNESS_THRESHOLD}, n={len(scores)})")

    assert score >= FAITHFULNESS_THRESHOLD, (
        f"Faithfulness {score:.3f} is below threshold {FAITHFULNESS_THRESHOLD}. "
        "The model may be hallucinating beyond the retrieved wilderness medicine context."
    )


def test_context_precision():
    """
    Context Precision: are the retrieved chunks actually relevant to the query?
    A low score means retrieval is pulling noisy or irrelevant chunks.
    """
    check_google_key()
    records = load_dataset()

    print(f"\nRunning pipeline on {len(records)} questions...")
    rows = run_pipeline(records)

    scores = []
    for i, row in enumerate(rows):
        s = score_context_precision_one(row["question"], row["contexts"])
        print(f"  [{i+1}/{len(rows)}] context_precision={s:.2f}")
        scores.append(s)

    score = sum(scores) / len(scores)
    print(f"\nContext Precision score: {score:.3f}  (threshold: {CONTEXT_PRECISION_THRESHOLD}, n={len(scores)})")

    assert score >= CONTEXT_PRECISION_THRESHOLD, (
        f"Context Precision {score:.3f} is below threshold {CONTEXT_PRECISION_THRESHOLD}. "
        "Consider adjusting chunk size, overlap, or top-k retrieval count."
    )
