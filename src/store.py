from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    An in-memory vector store for text chunks.

    ChromaDB is intentionally not used: no test requires it, it is not in
    requirements.txt, and initializing it here (setting a "using chroma" flag
    before the client actually exists) would make every method branch into an
    unimplemented code path and fail if chromadb ever happened to be installed
    on the grading machine. In-memory is the only backend.

    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """
        Turn a Document into a stored record.

        Copies metadata (never mutate the caller's dict) and guarantees a
        doc_id key: delete_document() filters on metadata["doc_id"], and at
        CP5 several chunks share one source file with ids like "file#0",
        "file#1" — doc_id must point at the source file, not at the chunk id.
        setdefault only falls back to doc.id when the caller hasn't already
        set a file-level doc_id in metadata.
        """
        metadata = dict(doc.metadata)
        metadata.setdefault("doc_id", doc.id)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """
        Rank an arbitrary candidate set by similarity to query.

        search() and search_with_filter() differ only in which records they
        pass in here, so both always score/sort/trim the same way. The raw
        embedding vector is dropped from each result — it's only needed to
        compute the score, and leaving it in would dump a multi-hundred-float
        vector into every printed/logged result.
        """
        query_embedding = self._embedding_fn(query)
        ranked = []
        for record in records:
            result = {key: value for key, value in record.items() if key != "embedding"}
            result["score"] = _dot(query_embedding, record["embedding"])
            ranked.append(result)
        ranked.sort(key=lambda record: record["score"], reverse=True)
        return ranked[: max(0, top_k)]

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document's content and append it to the in-memory store."""
        self._store.extend(self._make_record(doc) for doc in docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query."""
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)
        candidates = [
            record
            for record in self._store
            if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        original_size = len(self._store)
        self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        return len(self._store) < original_size
