"""Build the Day 7 benchmark from the Day 6 questions plus new Day 7 cases.

- q01-q20: Day 6 benchmark (same split). Day 6 test was only reported, never used to tune.
- q21-q23: new answerable TEST questions written for Day 7 before any Day 7 experiment.
- u01-u04: unanswerable (near-topic) questions, 2 dev + 2 test; absence verified by
  the regex checks in ABSENCE_CHECKS over the full corpus (not only top-k).

Outputs data/benchmark/questions.json (answerable, gold evidence) and
data/benchmark/cases.json (all cases, answerable flag, no gold -> safe for agent runs).
Run: python -m src.build_benchmark
"""
import json
import re

from .config import CASES_PATH, CHUNKS_PATH, QUESTIONS_PATH, ROOT
from .evaluation import covers

# Day 6 q11 quotes a rule (khoản 2c, chunk s00) and its exception (khoản 3, chunk s01) that
# Day 6 chunking split in two. Day 7 keeps the question but makes it multi-evidence: both
# pieces are required for "đủ evidence".
MULTI_EVIDENCE = {
    "q11": [{"pages": [45], "evidence": "Bảo đảm số giờ làm thêm của người lao động không quá 200 giờ trong 01 năm, trừ trường hợp quy định tại khoản 3 Điều này."},
            {"pages": [45], "evidence": "Người sử dụng lao động được sử dụng người lao động làm thêm không quá 300 giờ trong 01 năm trong một số ngành, nghề, công việc hoặc trường hợp sau đây"}],
}

NEW_ANSWERABLE = [
    {"id": "q21", "split": "test", "type": "dien_dat_lai", "dieu": 113, "gold_page": [47],
     "question": "Làm đủ một năm cho một công ty thì người lao động bình thường được nghỉ phép năm bao nhiêu ngày?",
     "expected_answer": "12 ngày làm việc, hưởng nguyên lương (công việc trong điều kiện bình thường).",
     "evidence_quote": "a) 12 ngày làm việc đối với người làm công việc trong điều kiện bình thường;",
     "filter": None, "notes": "Day 7 mới; chương VII có câu dev (Điều 105, 107, 112, 115) nhưng khác Điều."},
    {"id": "q22", "split": "test", "type": "dieu_kien_ngoai_le", "dieu": 128, "gold_page": [53],
     "question": "Người lao động bị tạm đình chỉ công việc tối đa bao lâu và trong thời gian đó có được ứng lương không?",
     "expected_answer": "Không quá 15 ngày, trường hợp đặc biệt không quá 90 ngày; được tạm ứng 50% tiền lương trước khi bị đình chỉ.",
     "evidence_quote": "Thời hạn tạm đình chỉ công việc không được quá 15 ngày, trường hợp đặc biệt không được quá 90 ngày. Trong thời gian bị tạm đình chỉ công việc, người lao động được tạm ứng 50% tiền lương trước khi bị đình chỉ công việc.",
     "filter": None, "notes": "Day 7 mới; chương VIII có câu dev (Điều 123, 125) nhưng khác Điều."},
    {"id": "q23", "split": "test", "type": "truc_tiep", "dieu": 139, "gold_page": [57],
     "question": "Lao động nữ được nghỉ thai sản bao lâu, trước khi sinh được nghỉ tối đa bao nhiêu?",
     "expected_answer": "06 tháng trước và sau khi sinh; nghỉ trước khi sinh không quá 02 tháng (sinh đôi trở lên: thêm 01 tháng mỗi con từ con thứ 2).",
     "evidence_quote": "Lao động nữ được nghỉ thai sản trước và sau khi sinh con là 06 tháng; thời gian nghỉ trước khi sinh không quá 02 tháng.",
     "filter": None, "notes": "Day 7 mới; chương X có câu dev q14 (Điều 137) nhưng khác Điều."},
]

UNANSWERABLE = [
    {"id": "u01", "split": "dev",
     "question": "Mức lương tối thiểu vùng I hiện nay là bao nhiêu đồng một tháng?",
     "absence_reason": "Điều 91 chỉ nói mức lương tối thiểu xác lập theo vùng và do Chính phủ công bố; bộ luật không ghi số tiền nào."},
    {"id": "u02", "split": "dev",
     "question": "Công ty không ký hợp đồng lao động bằng văn bản với người lao động thì bị phạt tiền bao nhiêu?",
     "absence_reason": "Mức phạt hành chính nằm ở nghị định xử phạt, không có trong bộ luật; 'phạt tiền' chỉ xuất hiện ở Điều 127 (cấm phạt tiền thay kỷ luật)."},
    {"id": "u03", "split": "test",
     "question": "Người lao động phải đóng bảo hiểm xã hội bắt buộc bao nhiêu phần trăm tiền lương mỗi tháng?",
     "absence_reason": "Điều 168 dẫn chiếu pháp luật về bảo hiểm xã hội; không có tỷ lệ đóng trong bộ luật. Bẫy gần chủ đề: Điều 219 (sửa Luật BHXH) có '20 năm đóng bảo hiểm xã hội' và '61%' (tỷ lệ suy giảm khả năng lao động), không phải tỷ lệ đóng."},
    {"id": "u04", "split": "test",
     "question": "Người bị mất việc được hưởng trợ cấp thất nghiệp tối đa bao nhiêu tháng?",
     "absence_reason": "Trợ cấp thất nghiệp thuộc Luật Việc làm; bộ luật chỉ nhắc tham gia bảo hiểm thất nghiệp, không có thời gian hưởng."},
]

# Corpus-wide checks documenting why each unanswerable case has no evidence.
ABSENCE_CHECKS = {
    "u01": r"\d[\d.]*\s*(đồng|triệu)",
    "u02": r"phạt tiền\s+(từ\s+)?\d",
    "u03": r"(mức|tỷ lệ) đóng[^.]{0,80}\d+\s*%",
    "u04": r"trợ cấp thất nghiệp",
}


def load_day6():
    rows = [json.loads(l) for l in (ROOT / "data" / "day6_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def to_day7(q):
    if q["id"] in MULTI_EVIDENCE:
        gold = [{"source_id": "blld-2019", "all_of": MULTI_EVIDENCE[q["id"]]}]
    else:
        gold = [{"source_id": q.get("gold_source_id", "blld-2019"), "pages": q["gold_page"], "evidence": q["evidence_quote"]}]
    gold += [{"source_id": "blld-2019", "pages": a["gold_page"], "evidence": a["evidence_quote"]} for a in q.get("alternatives", [])]
    return {"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
            "expected_answer": q["expected_answer"], "gold": gold, "filter": q.get("filter"),
            "dieu": q.get("dieu"), "notes": q.get("notes", ""),
            "multi_evidence": q["id"] in MULTI_EVIDENCE,
            "origin": "day6" if q["id"] <= "q20" else "day7"}


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    questions = [to_day7(q) for q in load_day6() + NEW_ANSWERABLE]
    for q in questions:   # every gold evidence must sit inside one chunk on the right page range
        if not covers(chunks, q["gold"]):
            raise ValueError(f"{q['id']}: evidence not found inside a single chunk")
    for u in UNANSWERABLE:
        hits = [c["id"] for c in chunks if re.search(ABSENCE_CHECKS[u["id"]], c["text"], re.I)]
        if hits:
            raise ValueError(f"{u['id']}: absence check matched {hits}")
        u["absence_check_regex"] = ABSENCE_CHECKS[u["id"]]
    cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "filter": q["filter"],
              "answerable": True, "expected_behavior": "search_and_answer"} for q in questions]
    cases += [{**u, "filter": None, "answerable": False, "expected_behavior": "search_and_abstain"} for u in UNANSWERABLE]
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for split in ("dev", "test"):
        a = sum(q["split"] == split for q in questions)
        u = sum(c["split"] == split and not c["answerable"] for c in cases)
        print(f"{split}: {a} answerable + {u} unanswerable")


if __name__ == "__main__":
    main()
