"""
RAGAS evaluation — faithfulness + context precision.

Uses OpenAI as the judge model (cloud, eval-only).
The production pipeline (gemma3:4b + ChromaDB) remains fully local.

Prerequisites:
  1. PDFs ingested: make ingest
  2. Eval dataset generated: make eval-generate
  3. OPENAI_API_KEY set in .env

Run with: make eval
"""

import json
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.eval

DATASET_PATH = Path(__file__).parent / "eval_dataset.jsonl"

FAITHFULNESS_THRESHOLD = 0.75   # min acceptable faithfulness score
CONTEXT_PRECISION_THRESHOLD = 0.70  # min acceptable context precision score


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        pytest.skip(
            f"Eval dataset not found at {DATASET_PATH}. "
            "Run `make eval-generate` first."
        )
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_google_key():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set in .env — skipping RAGAS eval.")


def build_ragas_dataset(records: list[dict]) -> "Dataset":
    """
    Run each question through the live RAG + LLM pipeline, collect
    (question, contexts, answer, ground_truth) tuples, return as HF Dataset.
    """
    from datasets import Dataset
    from llm.ollama_client import build_messages, chat
    from rag.query import rag_engine

    rows = {"question": [], "contexts": [], "answer": [], "ground_truth": []}

    for record in records:
        question = record["question"]
        ground_truth = record["ground_truth"]

        # Live RAG retrieval
        context_chunks = rag_engine.retrieve(question)

        # Live LLM response
        messages = build_messages(question, context_chunks)
        answer = chat(messages, stream=False)

        rows["question"].append(question)
        rows["contexts"].append(context_chunks)
        rows["answer"].append(answer)
        rows["ground_truth"].append(ground_truth)

    return Dataset.from_dict(rows)


def get_gemini_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    # thinking_budget=0 disables Gemini 2.5-flash's chain-of-thought tokens so
    # RAGAS's JSON score parser can find the answer in the response.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model_kwargs={"thinking_config": {"thinking_budget": 0}},
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))
    return llm, embeddings


def test_faithfulness():
    """
    Faithfulness: the generated answer should not contradict or go beyond
    the retrieved context chunks. This is the core safety metric.
    """
    check_google_key()
    from ragas import evaluate
    from ragas.metrics import faithfulness

    records = load_dataset()
    dataset = build_ragas_dataset(records)
    llm, embeddings = get_gemini_llm()

    result = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=llm,
        embeddings=embeddings,
    )

    raw = result["faithfulness"]
    score = float(sum(raw) / len(raw)) if isinstance(raw, list) else float(raw)
    print(f"\nFaithfulness score: {score:.3f} (threshold: {FAITHFULNESS_THRESHOLD})")

    assert score >= FAITHFULNESS_THRESHOLD, (
        f"Faithfulness {score:.3f} is below threshold {FAITHFULNESS_THRESHOLD}. "
        "The model may be hallucinating beyond the retrieved wilderness medicine context."
    )


def test_context_precision():
    """
    Context Precision: are the retrieved chunks actually relevant to the query?
    A low score means the retrieval is pulling noisy/irrelevant chunks.
    """
    check_google_key()
    from ragas import evaluate
    from ragas.metrics import context_precision

    records = load_dataset()
    dataset = build_ragas_dataset(records)
    llm, embeddings = get_gemini_llm()

    result = evaluate(
        dataset,
        metrics=[context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    raw = result["context_precision"]
    score = float(sum(raw) / len(raw)) if isinstance(raw, list) else float(raw)
    print(f"\nContext Precision score: {score:.3f} (threshold: {CONTEXT_PRECISION_THRESHOLD})")

    assert score >= CONTEXT_PRECISION_THRESHOLD, (
        f"Context Precision {score:.3f} is below threshold {CONTEXT_PRECISION_THRESHOLD}. "
        "Consider adjusting chunk size, overlap, or top-k retrieval count."
    )
