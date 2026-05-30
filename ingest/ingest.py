"""
PDF ingestion pipeline.

Reads all PDFs from data/pdfs/, chunks them, embeds with nomic-embed-text
via Ollama, and stores in a persistent ChromaDB collection.

Run once (or re-run to add new documents):
    python -m ingest.ingest
"""

import os
import sys
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from rich.console import Console
from rich.progress import track

console = Console()

PDFS_DIR = Path(__file__).parent.parent / "data" / "pdfs"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "wilderness_medicine"

# Embedding model — nomic-embed-text runs in Ollama, same binary on Mac + Jetson
EMBED_MODEL = "nomic-embed-text"

# Chunk settings tuned for dense medical text
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def get_chroma_collection() -> tuple[chromadb.Collection, ChromaVectorStore]:
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return collection, vector_store


def ingest():
    if not PDFS_DIR.exists() or not any(PDFS_DIR.glob("*.pdf")):
        console.print(f"[yellow]No PDFs found in {PDFS_DIR}[/yellow]")
        console.print("Add wilderness medicine PDFs to data/pdfs/ and re-run.")
        sys.exit(0)

    pdf_files = list(PDFS_DIR.glob("*.pdf"))
    console.print(f"[green]Found {len(pdf_files)} PDF(s) to ingest:[/green]")
    for f in pdf_files:
        console.print(f"  • {f.name}")

    console.print(f"\n[cyan]Loading embedding model ({EMBED_MODEL}) via Ollama...[/cyan]")
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    console.print("[cyan]Reading and chunking PDFs...[/cyan]")
    reader = SimpleDirectoryReader(input_dir=str(PDFS_DIR), required_exts=[".pdf"])
    documents = reader.load_data()
    console.print(f"  Loaded {len(documents)} document chunks.")

    console.print("[cyan]Connecting to ChromaDB...[/cyan]")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    collection, vector_store = get_chroma_collection()

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    console.print("[cyan]Embedding and storing in ChromaDB (this may take a while)...[/cyan]")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    count = collection.count()
    console.print(f"\n[bold green]Done! {count} chunks stored in ChromaDB collection '{COLLECTION_NAME}'.[/bold green]")


if __name__ == "__main__":
    ingest()
