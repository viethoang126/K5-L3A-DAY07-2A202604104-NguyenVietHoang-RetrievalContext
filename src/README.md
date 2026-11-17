# Code Day 7: Agentic RAG trên Bộ luật Lao động 2019

| File | Công dụng |
|---|---|
| `config.py` | Đường dẫn, model, đọc `.env` (ở repo hoặc thư mục cha) |
| `export_day6.py` | Xuất 278 chunk từ Chroma Day 6 (`day06_blld_B`) sang `data/corpus/chunks.json` |
| `build_benchmark.py` | Tạo `data/benchmark/questions.json` (gold) và `cases.json` (không gold, dùng cho agent) |
| `evaluation.py` | Định nghĩa hit: đúng nguồn, trang nằm trong `[page, page_end]`, evidence nằm trọn trong chunk; hỗ trợ câu nhiều evidence (`all_of`) |
| `retrievers.py` | Dense (Chroma + e5-small), BM25 (âm tiết ± bigram), Hybrid RRF, Rerank cross-encoder |
| `context.py` | Dedup, đoạn lân cận cùng Điều, đóng gói theo ngân sách ký tự |
| `eval_retrieval.py` | Benchmark retrieval/context không dùng LLM (Hit@1, Hit@3, MRR, Context Evidence Rate) |
| `llm.py` | Client Gemini qua endpoint OpenAI-compatible (stdlib), retry, dừng khi hết quota ngày |
| `agent.py` | Vòng lặp agent: `search_law`, `submit_answer`, `abstain`; trace từng bước |
| `run_agent_bench.py` | Chạy agent trên một split (đọc `cases.json`, không có gold); chạy tiếp được nếu bị ngắt |
| `score_agent.py` | Chấm tự động sau khi chạy (dùng gold) và tổng hợp nhãn thủ công bằng `starter/agent_metrics.py` |
| `write_reviews.py` | Ghi nhãn chấm tay (`reviews.json`) kèm lý do |

## Cách chạy lại

Python 3.12, cài `requirements.txt`. Tạo `.env` từ `.env.example` và điền `GEMINI_API_KEY`. Nếu chạy trên Windows, đặt `PYTHONUTF8=1`.

```bash
python -m unittest discover -s tests -v              # 27 test, mock, không gọi API
python -m src.export_day6                            # (đã có sẵn data/corpus/chunks.json)
python -m src.build_benchmark
python -m src.eval_retrieval --split dev             # 7 cấu hình retrieval
python -m src.run_agent_bench --config A0_dense_single --split dev --model gemini-3.5-flash-lite
python -m src.score_agent --config A0_dense_single --split dev
python -m src.write_reviews --split dev              # nhãn thủ công -> reviews.json
python -m src.score_agent --config A0_dense_single --split dev   # thêm phần "manual"
# Sau khi chốt cấu hình (results/SELECTION_CRITERIA.md):
python -m src.eval_retrieval --split test --configs R1_dense R2b_bm25_unigram R3_hybrid R5_hybrid_rerank
python -m src.run_agent_bench --config A0_dense_single --split test --model gemini-3.5-flash-lite
```

Lần đầu chạy, model embedding `intfloat/multilingual-e5-small` và reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` được tải từ Hugging Face; từ lần sau đặt `HF_HUB_OFFLINE=1` để dùng bản cache.
