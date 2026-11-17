"""Write reviews.json from manual labels recorded below (one entry per case).

Labels were produced by reading each trace: the answer against expected_answer and the
FULL text of every cited chunk. Default for an answered case = grade 2 and every citation
supports its sentence; only deviations are listed in OVERRIDES with the reason.
Reviewer: Claude Code (AI-assisted) -> to be confirmed by the student.

Run: python -m src.write_reviews --root agent --split dev
"""
import argparse
import json

from .config import RESULTS_DIR

REVIEWER = "Claude Code (AI-assisted), cần sinh viên xác nhận"

# (root, split, config, query_id) -> overrides
OVERRIDES = {
    ("agent", "dev", "A1_dense_multi", "q08"): {
        "citation_support": {"blld-2019:B:d129:s00": False},
        "reason": "Đáp án 30% đúng, d102 hỗ trợ; d129 (Điều 129) chỉ nói bồi thường tối đa 03 tháng lương và dẫn sang khoản 3 Điều 102, không chứa con số 30% -> citation d129 không hỗ trợ câu được gắn."},
    ("agent", "dev", "A1_dense_multi", "q10"): {
        "answer_grade": 1,
        "reason": "Đúng mức nửa tháng lương/năm, đủ 12 tháng, 2 ngoại lệ; nhưng thiếu điều kiện chấm dứt theo các khoản 1, 2, 3, 4, 6, 7, 9, 10 Điều 34 (chỉ ghi 'chấm dứt theo quy định'). d040 hỗ trợ ý đơn phương trái luật không được trợ cấp."},
    ("agent", "dev", "A0_dense_single", "q11"): {
        "reason": "Đủ 200 giờ + ngoại lệ 300 giờ; d108 (Điều 108) hỗ trợ ý thêm về làm thêm không giới hạn trong trường hợp đặc biệt."},
    ("agent", "dev", "A2_rerank_multi", "q03"): {
        "reason": "Đúng 08 giờ/ngày, 48 giờ/tuần; thêm ý người chưa thành niên, d146 hỗ trợ đúng ý đó."},
    ("agent", "dev", "A2_rerank_multi", "q06"): {
        "reason": "Đúng 05 ngày/30 ngày hoặc 20 ngày/365 ngày; d036:s01 điểm e hỗ trợ ý thêm về đơn phương chấm dứt khi bỏ việc 05 ngày liên tục."},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="agent")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--configs", nargs="*")
    args = ap.parse_args()
    base = RESULTS_DIR / args.root / args.split
    for cfg_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if args.configs and cfg_dir.name not in args.configs:
            continue
        traces = json.loads((cfg_dir / "traces.json").read_text(encoding="utf-8"))
        reviews = []
        for t in traces:
            if t["status"] == "error":
                continue
            o = OVERRIDES.get((args.root, args.split, cfg_dir.name, t["query_id"]), {})
            sup = o.get("citation_support", {})
            r = {"query_id": t["query_id"], "reviewer": REVIEWER,
                 "citation_support": [sup.get(c, True) for c in t["citations"]],
                 "citations": t["citations"]}
            if t["answerable"]:
                r["answer_grade"] = 0 if t["status"] == "abstain" else o.get("answer_grade", 2)
                r["reason"] = o.get("reason", "Từ chối câu có đáp án." if t["status"] == "abstain"
                                    else "Đúng và đủ so với expected_answer; citation chứa thông tin được nêu.")
            else:
                r["abstention_correct"] = o.get("abstention_correct", t["status"] == "abstain")
                r["reason"] = o.get("reason", "Từ chối đúng, lý do nêu bộ luật không chứa con số được hỏi; không thêm khẳng định."
                                    if t["status"] == "abstain" else "Không từ chối câu thiếu bằng chứng.")
            reviews.append(r)
        (cfg_dir / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{cfg_dir.name}: {len(reviews)} reviews")


if __name__ == "__main__":
    main()
