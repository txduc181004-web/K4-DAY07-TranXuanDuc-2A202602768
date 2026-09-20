from __future__ import annotations

import re
import sys
from pathlib import Path

from src import Document, EmbeddingStore, KnowledgeBaseAgent, RecursiveChunker, _mock_embed

# CAVEAT: OpenAI (het credit) va Local sentence-transformers (het dung luong o
# C:) deu khong dung duoc luc chay benchmark nay -> dung MockEmbedder. Mock
# bam MD5, khong ma hoa ngu nghia, nen score/rank trong ket qua ben duoi KHONG
# phan anh do lien quan that. Trong tam phan tich phai chuyen sang count /
# avg_length / do mach lac chunk (xem ChunkingStrategyComparator) va sang
# honest_score (kiem noi dung marker) thay vi tin score cosine.


DATA_DIR = Path("data/shopee-returns")
CHUNK_SIZE = 500

QUERIES = [
    ("Q1", "Sau khi Shopee chấp nhận yêu cầu Trả hàng & Hoàn tiền, người mua phải gửi trả sản phẩm trong bao lâu?", None),
    ("Q2", "Với đơn hàng thanh toán bằng thẻ tín dụng/ghi nợ, người mua nhận tiền hoàn trong bao lâu?", None),
    ("Q3", "Khi tải bằng chứng cho yêu cầu Trả hàng/Hoàn tiền, dung lượng video tối đa được Shopee quy định là bao nhiêu?", None),
    ("Q4", "Những nhóm sản phẩm nào không áp dụng lý do trả hàng Đổi ý?", None),
    ("Q5", "Nếu không đồng ý với quyết định hoàn tiền, thời hạn phản hồi của người bán là bao lâu?", {"audience": "seller"}),
]

# Gold doc_id + một chuỗi đặc trưng (verify bằng grep là chỉ xuất hiện đúng 1 file
# trong corpus) phải có thật trong nội dung top-3 mới coi là "trả lời được" —
# kiểm ở mức nội dung, không chỉ kiểm doc_id có mặt trong top-3 hay không.
GOLD = {
    "Q1": {"doc_id": "shopee-return-process", "marker": "6 ngày kể từ thời điểm nhận được thông báo gửi trả hàng"},
    "Q2": {"doc_id": "shopee-refund-timeline", "marker": "7 - 14 ngày làm việc"},
    "Q3": {"doc_id": "shopee-return-evidence", "marker": "Không quá 100 MB/video"},
    "Q4": {"doc_id": "shopee-return-restrictions", "marker": "Sản phẩm số và dịch vụ"},
    "Q5": {"doc_id": "shopee-return-policy-seller", "marker": "2 ngày lịch"},
}


def score_query(results: list[dict], gold_doc_id: str, marker: str) -> dict:
    """
    Chấm 2 mức cho một câu hỏi, theo thang docs/SCORING.md (2/1/0 mỗi câu):
      - naive_score: chỉ kiểm doc_id của gold có nằm trong top-3 không, bất kể
        thứ hạng của nó có chunk nào thật sự chứa câu trả lời hay không. Đây là
        cách chấm ngây thơ, dễ bị thổi phồng (một chunker theo heading có thể
        chiếm trọn top-3 từ đúng tài liệu gold mà không chunk nào chứa đáp án).
      - honest_score: kiểm nội dung — chuỗi marker đặc trưng của gold answer
        phải thật sự xuất hiện trong text của top-3 chunk. 2đ nếu gold ở top-1
        và marker có mặt; 1đ nếu gold ở top-2/3 và marker có mặt; 0đ nếu gold
        vắng mặt hoặc marker không có trong ngữ cảnh (kể cả khi gold_doc_id có
        mặt nhưng đúng đoạn chứa đáp án lại không lọt top-3).
    """
    doc_order = [result["metadata"].get("doc_id") for result in results]
    gold_rank = doc_order.index(gold_doc_id) + 1 if gold_doc_id in doc_order else None
    context = "\n".join(result["content"] for result in results)
    marker_found = marker in context

    naive_score = 2 if gold_rank == 1 else 1 if gold_rank in (2, 3) else 0

    if not marker_found or gold_rank is None:
        honest_score = 0
    elif gold_rank == 1:
        honest_score = 2
    else:
        honest_score = 1

    return {
        "gold_rank": gold_rank,
        "marker_found": marker_found,
        "naive_score": naive_score,
        "honest_score": honest_score,
    }


def parse_markdown(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, re.DOTALL)
    if not match:
        return {}, text
    metadata = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, match.group(2).strip()


def load_chunk_documents() -> list[Document]:
    chunker = RecursiveChunker(chunk_size=CHUNK_SIZE)
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
    return documents


def demo_llm(prompt: str) -> str:
    """
    Khong co LLM chat that nao dung duoc trong moi truong nay (OpenAI het
    credit; embedder local/OpenAI chi la embedding, khong phai chat). Dung
    lai dung pattern demo_llm cua main.py: echo preview cua prompt, KHONG
    phai cau tra loi that duoc sinh ra - ghi ro trong report de khong hieu
    nham day la agent answer that.
    """
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM - khong phai cau tra loi that] {preview}..."


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    documents = load_chunk_documents()
    store = EmbeddingStore(collection_name="shopee_returns_recursive", embedding_fn=_mock_embed)
    store.add_documents(documents)
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"strategy=RecursiveChunker chunk_size={CHUNK_SIZE} embedder=mock chunks={store.get_collection_size()}")
    naive_total = honest_total = 0
    for query_id, query, metadata_filter in QUERIES:
        results = store.search_with_filter(query, top_k=3, metadata_filter=metadata_filter)
        print(f"\n{query_id}: {query}")
        print(f"filter={metadata_filter}")
        for rank, result in enumerate(results, start=1):
            print(f"  {rank}. score={result['score']:.4f} id={result['id']} doc_id={result['metadata'].get('doc_id')}")
            print(f"     {result['content'][:180].replace(chr(10), ' ')}")

        gold = GOLD[query_id]
        outcome = score_query(results, gold["doc_id"], gold["marker"])
        naive_total += outcome["naive_score"]
        honest_total += outcome["honest_score"]
        print(
            f"  scoring: gold_doc={gold['doc_id']} gold_rank={outcome['gold_rank']} "
            f"marker_found={outcome['marker_found']} "
            f"naive={outcome['naive_score']}/2 honest={outcome['honest_score']}/2"
        )
        print(f"  agent.answer(top_k=3) (top-3 co filter, demo_llm khong phai LLM that):")
        print(f"     {agent.answer(query, top_k=3, metadata_filter=metadata_filter)}")

    print(f"\n=== Tong ket cham diem ===")
    print(f"naive_total (chi kiem doc_id o top-3)  = {naive_total}/10")
    print(f"honest_total (kiem marker co trong context) = {honest_total}/10")
    if naive_total != honest_total:
        print(f"chenh lech = {naive_total - honest_total} diem -> naive thoi phong ket qua so voi honest")
    else:
        print("khong chenh lech giua 2 cach cham cho lan chay nay")


if __name__ == "__main__":
    main()