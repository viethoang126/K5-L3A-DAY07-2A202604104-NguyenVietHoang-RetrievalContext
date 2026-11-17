"""Retrieval + context benchmark (no LLM): original query, user filter, same corpus/index.

Metrics per config (answerable questions only):
  Hit@1, Evidence Hit@3 (always top-3, whatever top-k), MRR@10,
  Context Evidence Rate (gold evidence fully inside the packed context), context chars,
  median latency. Service cost is 0 (local models).

Run: python -m src.eval_retrieval --split dev
     python -m src.eval_retrieval --split test --configs R1_dense R7_selected
"""
import argparse
import json
import statistics
import time

from . import retrievers
from .config import CHUNKS_PATH, QUESTIONS_PATH, RESULTS_DIR
from .context import pack, with_neighbours
from .evaluation import covers, first_cover_rank, frac, gold_hit

# name -> (method, context top-k, context budget chars, neighbour expansion, candidates)
CONFIGS = {
    "R1_dense":          dict(method="dense",         k=3, budget=3000, neighbours=False),
    "R2_bm25":           dict(method="bm25",          k=3, budget=3000, neighbours=False),
    "R2b_bm25_unigram":  dict(method="bm25_unigram",  k=3, budget=3000, neighbours=False),
    "R3_hybrid":         dict(method="hybrid",        k=3, budget=3000, neighbours=False),
    "R4_dense_rerank":   dict(method="dense_rerank",  k=3, budget=3000, neighbours=False),
    "R5_hybrid_rerank":  dict(method="hybrid_rerank", k=3, budget=3000, neighbours=False),
    "R6_hybrid_rerank_k5_nb": dict(method="hybrid_rerank", k=5, budget=4500, neighbours=True),
}


def run_config(name, cfg, questions, chunks, cache):
    retr = retrievers.build(cfg["method"], chunks, candidates=cfg.get("candidates", 20), cache=cache)
    by_order = {c["metadata"]["order"]: c for c in chunks}
    traces = []
    for q in questions:
        t0 = time.perf_counter()
        results = retr.search(q["question"], filters=q.get("filter"), top_k=max(10, cfg["k"]))
        ctx_in = results[:cfg["k"]]
        if cfg["neighbours"]:
            ctx_in = with_neighbours(ctx_in, by_order)
        context, ids = pack(ctx_in, cfg["budget"])
        ms = (time.perf_counter() - t0) * 1000
        flags = [gold_hit(r, q["gold"]) for r in results]
        first = first_cover_rank(results, q["gold"])
        ctx_hit = covers([c for c in ctx_in if c["id"] in ids], q["gold"])
        traces.append({
            "query_id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
            "filter": q.get("filter"), "config": name, **cfg,
            "results": [{"rank": r["rank"], "id": r["id"], "score": round(r["score"], 5),
                         "page": r["page"], "page_end": r["metadata"].get("page_end"),
                         "chuong": r["metadata"].get("chuong"), "dieu": r["metadata"].get("dieu"),
                         "hit": f, **({"base_rank": r["base_rank"]} if "base_rank" in r else {})}
                        for r, f in zip(results, flags)],
            "hit1": first == 1, "hit3": first is not None and first <= 3,
            "rr10": 1 / first if first and first <= 10 else 0.0, "first_hit_rank": first,
            "context_ids": ids, "context_chars": len(context), "context_hit": ctx_hit,
            "elapsed_ms": round(ms, 1),
            "gold_dieu_for_analysis": q.get("dieu"),
        })
    n = len(traces)
    summary = {"config": name, **cfg, "split": questions[0]["split"], "N": n,
               "hit1": frac(sum(t["hit1"] for t in traces), n),
               "hit3": frac(sum(t["hit3"] for t in traces), n),
               "mrr10": round(sum(t["rr10"] for t in traces) / n, 3),
               "context_evidence_rate": frac(sum(t["context_hit"] for t in traces), n),
               "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
               "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1),
               "service_cost_usd": 0,
               "misses_hit3": [t["query_id"] for t in traces if not t["hit3"]],
               "misses_context": [t["query_id"] for t in traces if not t["context_hit"]]}
    return summary, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    questions = [q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")) if q["split"] == args.split]
    out = RESULTS_DIR / "retrieval" / args.split
    out.mkdir(parents=True, exist_ok=True)
    cache, summaries = {}, []
    for name in args.configs:
        s, traces = run_config(name, CONFIGS[name], questions, chunks, cache)
        summaries.append(s)
        (out / f"{name}.traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{name:26s} Hit@1 {s['hit1']['text']:6s} Hit@3 {s['hit3']['text']:6s} MRR {s['mrr10']:.3f} "
              f"Ctx {s['context_evidence_rate']['text']:6s} chars {s['avg_context_chars']:5d} "
              f"{s['median_ms']:7.1f} ms  miss@3 {s['misses_hit3']}")
    path = out / "summary.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    merged = {s["config"]: s for s in old} | {s["config"]: s for s in summaries}
    path.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
