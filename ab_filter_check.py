from __future__ import annotations

import sys
from pathlib import Path

from src import Document, EmbeddingStore, FixedSizeChunker, RecursiveChunker, SentenceChunker, _mock_embed
from bench import GOLD, parse_markdown

DATA_DIR = Path("data/shopee-returns")
CHUNK_SIZE = 500

# The one query in the shared benchmark that needs metadata_filter to reach
# its gold doc (see REPORT_NHOM.md muc 3, Q5).
FILTER_QUERY_ID = "Q5"
FILTER_QUERY = "Nếu không đồng ý với quyết định hoàn tiền, thời hạn phản hồi của người bán là bao lâu?"
FILTER = {"audience": "seller"}

STRATEGIES = {
    "fixed_size": lambda: FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=50),
    "by_sentences": lambda: SentenceChunker(max_sentences_per_chunk=3),
    "recursive": lambda: RecursiveChunker(chunk_size=CHUNK_SIZE),
}


def build_store(chunker) -> EmbeddingStore:
    store = EmbeddingStore(collection_name="ab_check", embedding_fn=_mock_embed)
    documents = []
    for path in sorted(DATA_DIR.glob("*.md")):
        metadata, content = parse_markdown(path)
        doc_id = metadata.get("doc_id", path.stem)
        for index, chunk in enumerate(chunker.chunk(content)):
            documents.append(
                Document(
                    id=f"{doc_id}#{index}",
                    content=chunk,
                    metadata={**metadata, "doc_id": doc_id, "chunk_index": index},
                )
            )
    store.add_documents(documents)
    return store


def top3_doc_ids(store: EmbeddingStore, metadata_filter: dict | None) -> list[str]:
    results = store.search_with_filter(FILTER_QUERY, top_k=3, metadata_filter=metadata_filter)
    return [result["metadata"].get("doc_id") for result in results]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gold_doc_id = GOLD[FILTER_QUERY_ID]["doc_id"]
    print(f"A/B check cho {FILTER_QUERY_ID} (can filter={FILTER}), gold_doc={gold_doc_id}\n")

    for name, make_chunker in STRATEGIES.items():
        store = build_store(make_chunker())
        no_filter = top3_doc_ids(store, None)
        with_filter = top3_doc_ids(store, FILTER)

        print(f"[{name}] chunks={store.get_collection_size()}")
        print(f"  khong filter : {no_filter}")
        print(f"  co filter    : {with_filter}")

        if no_filter == with_filter:
            print("  => GIONG HET NHAU: filter khong doi gi ca cho chien luoc nay -> can xem lai cau hoi/cach tach audience")
        elif gold_doc_id in with_filter and gold_doc_id not in no_filter:
            print(f"  => KHAC NHAU: filter dua {gold_doc_id} vao top-3 (khong filter thi khong co) -> filter co gia tri that")
        elif gold_doc_id in with_filter and gold_doc_id in no_filter:
            print(f"  => KHAC NHAU nhung {gold_doc_id} da co san trong top-3 khong filter -> filter khong lam thay doi ket luan cho query nay o chien luoc nay")
        else:
            print(f"  => KHAC NHAU nhung {gold_doc_id} van khong co trong top-3 du co filter -> can xem lai chunking/gold marker")
        print()


if __name__ == "__main__":
    main()
