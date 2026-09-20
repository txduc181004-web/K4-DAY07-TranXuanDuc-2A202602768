from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        results = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        if not results:
            return "No relevant context was found in the knowledge base to answer this question."

        context = "\n\n".join(
            f"[{index}] (source: {result['metadata'].get('doc_id', 'unknown')}) {result['content']}"
            for index, result in enumerate(results, start=1)
        )
        prompt = (
            "Answer the question using only the context below. "
            "Cite the bracketed source number(s), e.g. [1], for every claim in your answer, "
            "so the answer can be traced back to the exact chunk and source it came from. "
            "If the context does not contain the answer, say plainly that it is not available "
            "in the provided sources — do not invent or assume information that isn't there.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return self.llm_fn(prompt)
