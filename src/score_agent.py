"""Post-hoc scoring of agent traces with gold (gold is used ONLY here, after the run).

Automatic: first-search Hit@1/Hit@3, Context Evidence Rate on the final context,
search-count distribution, tokens, time, estimated cost.
Manual (from reviews.json, written by a human reviewer after reading each trace):
grounded answer 0/1/2, citation support, abstention — aggregated with starter/agent_metrics.py.

Run: python -m src.score_agent --config A0_dense_single --split dev
"""
import argparse
import json
import statistics
import sys
from collections import Counter

from .config import CASES_PATH, CHUNKS_PATH, QUESTIONS_PATH, RESULTS_DIR, ROOT
from .evaluation import covers, first_cover_rank, frac

sys.path.insert(0, str(ROOT / "starter"))
from agent_metrics import aggregate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    args = ap.parse_args()
    out = RESULTS_DIR / "agent" / args.split / args.config
    traces = json.loads((out / "traces.json").read_text(encoding="utf-8"))
    chunks = {c["id"]: c for c in json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))}
    gold = {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}
    cases = [c for c in json.loads(CASES_PATH.read_text(encoding="utf-8")) if c["split"] == args.split]
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    per_case = []
    for t in ok:
        row = {"query_id": t["query_id"], "status": t["status"], "n_searches": t["n_searches"],
               "citations": t["citations"], "stop_reason": t.get("stop_reason")}
        if t["answerable"]:
            g = gold[t["query_id"]]["gold"]
            searches = [e for e in t["events"] if e["event"] == "search"]
            first = [chunks[r["id"]] for r in searches[0]["results"]] if searches else []
            rank = first_cover_rank(first, g)
            row |= {"first_search_hit1": rank == 1, "first_search_hit3": rank is not None and rank <= 3,
                    "context_has_evidence": covers([chunks[i] for i in t["context_ids"]], g),
                    "cited_chunks_cover_evidence": covers([chunks[i] for i in t["citations"]], g)}
        per_case.append(row)
    usage = [t.get("usage") or {} for t in ok]
    summary = {
        "config": args.config, "split": args.split, "model": traces[0].get("model") if traces else None,
        "served_models": sorted({e.get("served_model") or "?" for t in ok for e in t["events"] if e["event"] == "llm_call"}),
        "cases": len(traces), "provider_errors": len(traces) - len(ok),
        "first_search_hit1": frac(sum(r.get("first_search_hit1", False) for r in per_case), len(ans)),
        "first_search_hit3": frac(sum(r.get("first_search_hit3", False) for r in per_case), len(ans)),
        "context_evidence_rate": frac(sum(r.get("context_has_evidence", False) for r in per_case), len(ans)),
        "cited_chunks_cover_evidence_auto": frac(sum(r.get("cited_chunks_cover_evidence", False) for r in per_case), len(ans)),
        "searches_distribution": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
        "mean_searches": round(statistics.mean(t["n_searches"] for t in ok), 2) if ok else None,
        "llm_calls": sum(u.get("llm_calls", 0) for u in usage),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
        "median_elapsed_s": statistics.median(t["elapsed_s"] for t in ok) if ok else None,
        "cost_usd_if_paid": round(sum(t.get("cost_usd_if_paid", 0) for t in traces), 5),
        "cost_usd_actual": 0.0,
        "avg_context_chars": round(statistics.mean(len(t["final_context"]) for t in ok)) if ok else None,
        "auto_abstention": frac(sum(t["status"] == "abstain" for t in ok if not t["answerable"]),
                                sum(not t["answerable"] for t in ok)),
        "auto_false_abstention": frac(sum(t["status"] == "abstain" for t in ans), len(ans)),
    }
    reviews_path = out / "reviews.json"
    if reviews_path.exists():
        summary["manual"] = aggregate(cases, traces, json.loads(reviews_path.read_text(encoding="utf-8")))
    else:
        summary["manual"] = "chưa có reviews.json (cần chấm thủ công)"
    (out / "per_case_auto.json").write_text(json.dumps(per_case, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
