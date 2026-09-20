# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Xuân Đức
**Nhóm:** TooSweet
**Ngày:** 20/9/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding có góc giữa chúng gần 0°, tức là hướng ngữ nghĩa gần như trùng nhau — hai đoạn văn bản mang ý nghĩa tương tự nhau, bất kể độ dài vector hay từ ngữ cụ thể dùng để diễn đạt.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Người mua có thể yêu cầu hoàn tiền nếu sản phẩm nhận được bị lỗi."
- Câu B: "Nếu hàng hóa giao đến có khiếm khuyết, khách hàng được phép đề nghị trả lại tiền."
- Tại sao tương đồng: Hai câu dùng từ vựng gần như khác hoàn toàn (mua/khách hàng, hoàn tiền/trả lại tiền, lỗi/khiếm khuyết, yêu cầu/đề nghị) nhưng cùng diễn đạt một ý: quyền được hoàn tiền khi sản phẩm lỗi. Embedding tốt phải nắm được nghĩa chứ không so khớp từ, nên độ tương tự cosine giữa hai câu này phải cao.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Người mua có thể yêu cầu hoàn tiền nếu sản phẩm nhận được bị lỗi."
- Câu B: "Người bán phải chuẩn bị và đóng gói đơn hàng trong vòng 2 ngày làm việc."
- Tại sao khác: Hai câu nói về hai chủ đề khác nhau trong cùng domain (hoàn tiền cho người mua vs. thời hạn chuẩn bị hàng của người bán) — không chia sẻ ý nghĩa cốt lõi, nên vector embedding của chúng lệch hướng nhau và cosine similarity thấp.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ đo góc (hướng ngữ nghĩa) giữa hai vector, không bị ảnh hưởng bởi độ lớn (magnitude) của vector — vốn có thể thay đổi theo độ dài câu hay tần suất từ mà không phản ánh sự khác biệt về ý nghĩa. Euclidean distance thì cộng dồn cả sai khác về độ lớn lẫn hướng, nên hai câu đồng nghĩa nhưng có độ dài/mật độ từ khác nhau vẫn có thể bị tính là "xa nhau" dù về mặt ngữ nghĩa chúng rất gần.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính: `ceil((length − overlap) / (chunk_size − overlap)) = ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
>
> Kiểm lại bằng `FixedSizeChunker` thật trong repo (`python -c "from src.chunking import FixedSizeChunker; print(len(FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)))"`) → kết quả in ra là **23**, khớp với công thức.
>
> Đáp án: **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Chạy lại với `overlap=100` trên cùng `FixedSizeChunker` cho kết quả **25 chunks** (tăng thêm 2 so với overlap=50), khớp công thức `ceil((10000−100)/(500−100)) = ceil(9900/400) = 25`. Overlap tăng làm bước trượt giữa các chunk (`chunk_size − overlap`) nhỏ lại (450 → 400 ký tự), nên cần nhiều chunk hơn để phủ hết tài liệu. Muốn overlap lớn hơn vì nó giảm rủi ro một câu/điều khoản/con số quan trọng bị cắt đúng ngay ranh giới giữa hai chunk — đổi lại là tốn thêm chunk (thêm chi phí lưu trữ và tính điểm truy xuất), nên cần cân bằng giữa độ an toàn ngữ cảnh và chi phí.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng regex `(?<=[.!?])(?:\s+|\n+)` với lookbehind: tách ở vị trí *sau* dấu `.`/`!`/`?` mà vẫn giữ được dấu đó ở cuối câu trước (nếu split bằng `[.!?]\s+` thông thường thì dấu câu bị nuốt vào phần khớp và mất khỏi câu). Sau khi tách, `strip()` từng câu và bỏ chuỗi rỗng, rồi gom mỗi `max_sentences_per_chunk` câu thành một chunk bằng `" ".join(...)`. Text rỗng/chỉ chứa khoảng trắng trả về `[]` ngay từ đầu, không crash.
>
> **Edge case chưa xử lý được** (biết trước, chưa fix): vì chỉ tách theo dấu câu + khoảng trắng nên gặp **chữ viết tắt** (`TS.`, `v.v.`) hoặc **số thập phân viết cách** (`3. 14`) sẽ bị coi là ranh giới câu sai. Verify thực tế: `SentenceChunker(1).chunk('TS. Nguyen la giao vien.')` → `['TS.', 'Nguyen la giao vien.']` (bị tách sai làm đôi); `SentenceChunker(1).chunk('Ho dung nhieu do dung v.v. de chuan bi.')` → `['Ho dung nhieu do dung v.v.', 'de chuan bi.']` (cũng tách sai). Số thập phân viết liền như `3.14` thì an toàn vì regex đòi hỏi có khoảng trắng ngay sau dấu `.` mới tách.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thử separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]` — ranh giới "to" (đoạn văn) trước để giữ ngữ nghĩa, chỉ hạ xuống ranh giới nhỏ hơn khi mảnh vẫn dài hơn `chunk_size`. Thuật toán có 2 chiều: (1) **đệ quy xuống** — nếu `current_text.split(separator)` không tách được gì (chỉ 1 mảnh) thì gọi lại `_split` với `remaining_separators[1:]`; (2) **gom lên** — các mảnh nhỏ liền kề được nối dần vào biến `pending` cho tới sát `chunk_size` rồi mới flush thành 1 chunk, tránh sinh chunk vụn vài ký tự. Có 3 base case dừng: mảnh đã `<= chunk_size` (trả nguyên); hết `remaining_separators` (cắt cứng theo `chunk_size`); và separator hiện tại là `""` (cắt cứng ngay, không cần `split("")`). Verify tay: `RecursiveChunker(separators=[], chunk_size=200).chunk('x'*1000)` → 5 chunk đúng 200 ký tự — base case "hết separator" xử lý đúng ngay cả khi `separators=[]` được truyền thẳng vào constructor (khớp `test_empty_separators_falls_back_gracefully`), không crash.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Chỉ dùng in-memory (`self._store`, một `list[dict]`), bỏ hẳn nhánh ChromaDB dù docstring gốc có nhắc tới — không test nào cần nó, `requirements.txt` không cài, và code khởi tạo cũ có bẫy thật: `self._use_chroma = True` được gán ngay khi `import chromadb` thành công, *trước khi* client/collection được tạo, nên nếu máy chấm bài tình cờ có `chromadb`, mọi method sẽ rẽ nhầm nhánh chưa implement và cả 14 test của `EmbeddingStore` sập theo. `add_documents` gọi helper `_make_record` (embed nội dung + copy metadata) cho từng `Document` rồi `extend` vào `self._store` — không tự chunk, 1 `Document` = 1 record. `search` gọi helper `_search_records` dùng `_dot` (vector đã chuẩn hoá nên dot product = cosine) để tính điểm cho toàn bộ `self._store`, sắp xếp giảm dần, cắt `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc trước rồi mới search**, không phải ngược lại. Nếu lấy `top_k` kết quả rồi mới lọc theo metadata, các slot `top_k` có thể đã bị tài liệu không khớp chiếm hết, trả về 0 kết quả dù store vẫn còn tài liệu hợp lệ. Verify bằng thực nghiệm: 5 tài liệu `department=sales` (điểm cao vì trùng từ khoá) + 1 tài liệu `department=engineering` (đúng nội dung cần tìm); lấy `top_k=2` trước rồi lọc `department=engineering` sau → **0 kết quả**; lọc `department=engineering` trước rồi mới search trong tập đã lọc → **1 kết quả** (`target`). Vì vậy `search_with_filter` xây `candidates` bằng list-comprehension lọc `self._store` theo `metadata_filter` trước, sau đó mới gọi `_search_records(query, candidates, top_k)` — dùng chung logic tính điểm với `search()` nên không thể lệch kết quả giữa hai đường code. `delete_document` xoá bằng cách giữ lại mọi record có `metadata['doc_id'] != doc_id` (list-comprehension mới), so sánh kích thước trước/sau để trả `True`/`False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> 3 nhịp: `store.search(question, top_k)` lấy top-k chunk → dựng prompt có ngữ cảnh → gọi `llm_fn(prompt)`. Nếu `results` rỗng (store rỗng hoặc không có kết quả), trả thẳng câu thông báo và **không gọi `llm_fn`** — verify tay: tạo store rỗng, đếm số lần `llm_fn` được gọi = 0, không có exception nào xảy ra.
>
> Ngữ cảnh được dựng dưới dạng `[{index}] (source: {doc_id}) {content}` cho từng chunk — vừa đánh số vừa gắn nguồn (`metadata['doc_id']`, trỏ về file gốc chứ không phải id của chunk nhờ `_make_record` ở `EmbeddingStore`). Prompt yêu cầu model **trích dẫn số ngoặc vuông** (`[1]`, `[2]`...) cho mọi khẳng định trong câu trả lời — đáp ứng tiêu chí *Source Traceability* ở `docs/EVALUATION.md`: câu trả lời truy vết được về đúng chunk và đúng file nguồn. Prompt cũng có ràng buộc chống bịa: chỉ dùng đúng ngữ cảnh được cung cấp, nếu không có thông tin thì nói rõ là "not available in the provided sources", không được suy đoán.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.20s ==============================

$ python main.py "Chunking là gì?"
=== Manual File Test ===
Accepted file types: .md, .txt
Input file list:
  - data/python_intro.txt
  - data/vector_store_notes.md
  - data/rag_system_design.md
  - data/customer_support_playbook.txt
  - data/chunking_experiment_report.md
  - data/vi_retrieval_notes.md
Skipping missing file: data\customer_support_playbook.txt   # binh thuong, repo khong co file nay

Loaded 5 documents
Embedding backend: mock embeddings fallback
Stored 5 documents in EmbeddingStore

=== EmbeddingStore Search Test ===
Query: Chunking là gì?
1. score=0.150 source=data\rag_system_design.md
2. score=0.027 source=data\python_intro.txt
3. score=0.025 source=data\chunking_experiment_report.md

=== KnowledgeBaseAgent Test ===
Question: Chunking là gì?
Agent answer:
[DEMO LLM] Generated answer from prompt preview: Answer the question using only the context below.
Cite the bracketed source number(s), e.g. [1], for every claim in your answer...
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Tính bằng `compute_similarity(_mock_embed(a), _mock_embed(b))` — dùng ngưỡng `>= 0.2` là "cao" (mốc đặt trên vùng nhiễu ~[-0.1, 0.1] quan sát được ở các cặp không liên quan).

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Người mua có thể yêu cầu hoàn tiền nếu sản phẩm bị lỗi." | "Khách hàng được phép đề nghị trả lại tiền khi hàng hóa có khiếm khuyết." (đồng nghĩa, khác từ vựng) | cao | 0.2336 | Đúng |
| 2 | "Người mua có thể yêu cầu hoàn tiền nếu sản phẩm bị lỗi." | "Hôm nay trời mưa to ở Hà Nội." (khác chủ đề hoàn toàn) | thấp | -0.0989 | Đúng |
| 3 | "Người mua có 7 ngày để gửi yêu cầu trả hàng." | "Người bán có 7 ngày để gửi yêu cầu trả hàng." (giống hệt câu chữ, chỉ đổi 1 từ mua→bán, nhưng đổi hẳn vai trò/đối tượng) | cao | -0.0340 | **Sai** |
| 4 | "Dung lượng video tối đa khi tải bằng chứng là 100 MB." | "Người bán phải phản hồi trong vòng 2 ngày lịch." (cùng miền chủ đề TMĐT nhưng khác nội dung cụ thể) | thấp | 0.0995 | Đúng |
| 5 | "Thời hạn hoàn tiền cho thẻ tín dụng là bao lâu?" | "Thẻ tín dụng/ghi nợ: 7-14 ngày làm việc." (cặp câu hỏi – câu trả lời trực tiếp) | cao | -0.0228 | **Sai** |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 5 — một câu hỏi và đúng câu trả lời trực tiếp cho nó — lại ra điểm **âm** (-0.0228), thấp hơn cả cặp 4 (hai câu hoàn toàn không liên quan tới nhau). Điều này không nói gì về "ý nghĩa" thật cả — nó lộ ra rằng `_mock_embed` không mã hoá ngữ nghĩa: vector được sinh từ hash MD5 rồi khuếch tán bằng LCG, nên giống nhau về ý nghĩa (kể cả quan hệ hỏi–đáp rõ ràng) không hề kéo hai vector lại gần nhau. Cặp 3 cũng sai theo đúng hướng đáng lo nhất: hai câu gần như giống hệt về mặt chữ (chỉ khác "mua"/"bán") nhưng khác hẳn về vai trò/đối tượng vẫn ra điểm thấp *tình cờ* đúng theo trực giác con người, trong khi với một embedding ngữ nghĩa thật thì nhiều khả năng nó sẽ cho điểm CAO (vì bề mặt từ vựng giống 90%) — đây chính xác là lý do `metadata_filter={"audience": ...}` cần tồn tại thay vì tin tưởng embedding tự phân biệt được buyer/seller, như đã chứng minh bằng A/B thật ở `REPORT_NHOM.md` mục 3.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

Chạy bằng `bench.py` (strategy=RecursiveChunker, chunk_size=500, embedder=mock — xem caveat mục 3), output đầy đủ lưu ở [`ket_qua_benchmark.txt`](../ket_qua_benchmark.txt). Cột "Có liên quan?" dùng tiêu chí **nội dung** (marker đáp án thật có mặt trong ngữ cảnh top-3), không phải chỉ kiểm `doc_id` — xem mục "Chấm hai mức" ở `REPORT_NHOM.md` mục 3. Cột "Câu trả lời của Agent" dùng `demo_llm` (echo lại đầu prompt) vì môi trường không có LLM chat thật khả dụng lúc này (OpenAI hết credit) — **không phải câu trả lời được sinh ra thật**, chỉ xác nhận `agent.answer()` chạy đúng luồng (kèm filter đúng cho Q5).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời hạn gửi trả sau khi được chấp nhận | `shopee-return-policy-buyer#8` — "...Shopee có thể quyết định hoàn tiền ngay mà không cần chờ trả sản phẩm..." | 0.3216 | Không (gold `shopee-return-process` ở hạng 2, marker "6 ngày..." không có trong top-3) | [DEMO LLM] echo prompt, không phải câu trả lời thật |
| 2 | Thời gian hoàn tiền thẻ tín dụng/ghi nợ | `shopee-return-policy-seller#2` — "Chi phí vận chuyển hoàn trả người bán phải chịu..." | 0.2781 | Không (gold `shopee-refund-timeline` không lọt top-3) | [DEMO LLM] echo prompt, không phải câu trả lời thật |
| 3 | Dung lượng video tối đa | `shopee-return-evidence#3` — đúng tài liệu, sai section (nói về video đóng gói, không phải giới hạn dung lượng) | 0.2728 | Không (gold doc ở top-1 nhưng marker "100 MB/video" nằm ở chunk #5, không lọt top-3 — xem failure case ở `REPORT_NHOM.md`) | [DEMO LLM] echo prompt, không phải câu trả lời thật |
| 4 | Nhóm sản phẩm hạn chế "Đổi ý" | `shopee-return-policy-seller#5` — "Liên lạc và tranh chấp..." | 0.2598 | Không (gold `shopee-return-restrictions` không lọt top-3) | [DEMO LLM] echo prompt, không phải câu trả lời thật |
| 5 | Thời hạn phản hồi của người bán | `shopee-return-policy-seller#6` — dòng ngày hiệu lực chính sách, không phải đoạn "2 ngày lịch" | 0.0616 | Không (gold doc ở top-1 nhờ `metadata_filter={"audience":"seller"}`, nhưng marker "2 ngày lịch" ở chunk khác, không lọt top-3) | [DEMO LLM] echo prompt, không phải câu trả lời thật |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **0 / 5** (theo tiêu chí nội dung/honest). Theo tiêu chí ngây thơ chỉ kiểm `doc_id` thì sẽ ra 2/5 (Q3, Q5 có gold doc ở top-1) — đúng khoảng cách 5 điểm (naive 5/10 vs honest 0/10) đã ghi trong `REPORT_NHOM.md` mục 3, và là lý do không nên chấm chỉ bằng `doc_id`.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu (cần xem demo của nhóm khác trước khi điền):*

---

## Tự Đánh Giá (Phần Cá Nhân)

> Điểm đề xuất dưới đây dựa trên mức độ hoàn thành có thể verify được (test pass, code chạy, có dẫn chứng thực nghiệm) — tự điều chỉnh nếu bạn đánh giá khác.

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 (42/42 test pass) |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 (đã chạy + phân tích trung thực, nhưng honest=0/5 vì đang dùng MockEmbedder — trừ nhẹ vì chưa verify được bằng embedder thật trong repo này) |
| **Tổng phần cá nhân** | **58 / 60** |
