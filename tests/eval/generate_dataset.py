"""
Synthetic eval dataset generator.

Pulls chunks from the ChromaDB index and uses OpenAI to generate
realistic wilderness medicine Q&A pairs from each chunk.

Output: tests/eval/eval_dataset.jsonl
Each line: {"question": "...", "ground_truth": "...", "ground_truth_context": "..."}

Usage:
    python -m tests.eval.generate_dataset [--n-chunks 40]

Run this once after ingestion, then commit eval_dataset.jsonl so the
RAGAS eval has a stable dataset to run against.
"""

import argparse
import json
import sys
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import track

load_dotenv()

console = Console()

CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"
COLLECTION_NAME = "wilderness_medicine"
OUTPUT_PATH = Path(__file__).parent / "eval_dataset.jsonl"

GENERATION_PROMPT = """\
You are a wilderness medicine expert creating a test dataset.

Below is an excerpt from a wilderness medicine reference manual.
Generate one realistic question that a rescuer in the field might ask,
along with a concise, accurate answer based ONLY on the text provided.

Rules:
- The question should be practical and field-relevant.
- The answer must be derivable from the context alone (no outside knowledge).
- Keep the answer under 100 words.
- Do not reference "the text" or "the passage" — write naturally.

Context:
{chunk}

Respond with valid JSON only, in this exact format:
{{"question": "...", "answer": "..."}}
"""


def generate_qa_pair(client: "genai.Client", chunk: str, retries: int = 3) -> dict | None:
    import time
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=GENERATION_PROMPT.format(chunk=chunk),
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            result = json.loads(response.text)
            return {
                "question": result["question"],
                "ground_truth": result["answer"],
                "ground_truth_context": chunk,
            }
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < retries - 1:
                # Extract retry delay from error if available, otherwise back off
                wait = 40 * (attempt + 1)
                console.print(f"[yellow]Rate limited — waiting {wait}s before retry {attempt + 2}/{retries}...[/yellow]")
                time.sleep(wait)
            elif "429" in err and "limit: 0" in err:
                console.print("[bold red]Free tier daily quota exhausted.[/bold red] Enable billing at aistudio.google.com or wait until tomorrow.")
                return None
            else:
                console.print(f"[yellow]Skipping chunk — generation failed: {e}[/yellow]")
                return None
    return None


def main(n_chunks: int = 40):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] GOOGLE_API_KEY not set in .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = db.get_collection(COLLECTION_NAME)
    except Exception:
        console.print(f"[bold red]Error:[/bold red] Collection '{COLLECTION_NAME}' not found. Run ingestion first.")
        sys.exit(1)

    total = collection.count()
    if total == 0:
        console.print("[yellow]ChromaDB is empty. Run ingestion first.[/yellow]")
        sys.exit(1)

    console.print(f"[green]{total} chunks in index. Sampling {n_chunks} for eval dataset...[/green]")

    # Sample evenly across the collection to cover the whole corpus
    results = collection.get(limit=n_chunks, include=["documents"])
    chunks = results["documents"]

    pairs = []

    for chunk in track(chunks, description="Generating Q&A pairs via Gemini 2.0 Flash..."):
        if len(chunk.strip()) < 100:
            continue  # skip very short chunks (headers, page numbers, etc.)
        pair = generate_qa_pair(client, chunk)
        if pair:
            pairs.append(pair)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    console.print(f"\n[bold green]Done! {len(pairs)} Q&A pairs written to {OUTPUT_PATH}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-chunks", type=int, default=40,
                        help="Number of chunks to sample from ChromaDB (default: 40)")
    args = parser.parse_args()
    main(n_chunks=args.n_chunks)
