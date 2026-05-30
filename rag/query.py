"""
RAG query interface.

Loads the existing ChromaDB index and retrieves the top-k most relevant
chunks for a given query. Returns plain text snippets ready to inject
into the LLM prompt.
"""

from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding


CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "wilderness_medicine"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 5  # number of chunks to retrieve per query


class RAGEngine:
    def __init__(self):
        self._index: VectorStoreIndex | None = None

    def _load(self):
        """Lazy-load the index on first query."""
        embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
        Settings.embed_model = embed_model

        db = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        """
        Retrieve the top-k most relevant text chunks for a query.

        Returns:
            List of text strings (document snippets), or empty list if
            the index has no documents yet.
        """
        if self._index is None:
            self._load()

        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        return [node.get_content() for node in nodes]

    def is_empty(self) -> bool:
        """Return True if ChromaDB has no documents ingested yet."""
        try:
            db = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = db.get_or_create_collection(COLLECTION_NAME)
            return collection.count() == 0
        except Exception:
            return True


# Module-level singleton — instantiated once, reused across queries
rag_engine = RAGEngine()
