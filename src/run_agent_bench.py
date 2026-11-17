"""Run the real agent over a split. Reads only data/benchmark/cases.json (question, user
filter, answerable flag for bookkeeping) -> no gold evidence/answer reaches the agent.

Resumable: cases already in the trace file with a non-error status are skipped, so a
rate-limit interruption can be continued by re-running the same command.

Run: python -m src.run_agent_bench --config A0_dense_single --split dev
"""
import argparse
import csv
import json

from . import retrievers
from .agent import run_agent
from .config import CASES_PATH, CHUNKS_PATH, RESULTS_DIR, ROOT
from .llm import GeminiClient, LLMError, QuotaExhausted

AGENT_CONFIGS = {
    # baseline: single dense search (classic RAG through the agent interface)
    "A0_dense_single":   dict(method="dense",         top_k=3, max_searches=1, budget=3000, neighbours=False),
    # improvement 1: let the model re-query (query rewriting / decomposition), same retriever
    "A1_dense_multi":    dict(method="dense",         top_k=3, max_searches=3, budget=4500, neighbours=False),
    # improvement 2: better ranking (hybrid RRF + cross-encoder) with multi-search
    "A2_rerank_multi":   dict(method="hybrid_rerank", top_k=3, max_searches=3, budget=4500, neighbours=False),
}


def load_chapters():
    with open(ROOT / "data" / "manifest_chapters.csv", encoding="utf-8") as f:
        return {r["chuong"]: r["title"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(AGENT_CONFIGS), required=True)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--ids", nargs="*", help="only these case IDs (debugging)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = AGENT_CONFIGS[args.config]
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    by_order = {c["metadata"]["order"]: c for c in chunks}
    cases = [c for c in json.loads(CASES_PATH.read_text(encoding="utf-8")) if c["split"] == args.split]
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    out = RESULTS_DIR / "agent" / args.split / args.config
    out.mkdir(parents=True, exist_ok=True)
    path = out / "traces.json"
    traces = {t["query_id"]: t for t in (json.loads(path.read_text(encoding="utf-8")) if path.exists() else [])}
    client = GeminiClient(model=args.model)
    retriever = retrievers.build(cfg["method"], chunks)
    chapters = load_chapters()
    for case in cases:
        if case["id"] in traces and traces[case["id"]]["status"] != "error":
            continue
        try:
            r = run_agent(case["question"], client, retriever, by_order=by_order, chapters=chapters,
                          filters=case.get("filter"), top_k=cfg["top_k"], max_searches=cfg["max_searches"],
                          budget=cfg["budget"], neighbours=cfg["neighbours"])
        except QuotaExhausted as e:
            print(f"{case['id']}: daily quota exhausted -> stopping; re-run the same command later to resume.\n{e}", flush=True)
            break
        except LLMError as e:
            r = {"status": "error", "error": str(e), "answer": None, "citations": [], "events": [],
                 "context_ids": [], "final_context": "", "n_searches": 0, "usage": {}, "elapsed_s": None}
        u = r.get("usage") or {}
        r["cost_usd_if_paid"] = round(client.cost_usd(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)), 6)
        traces[case["id"]] = {"query_id": case["id"], "split": case["split"], "question": case["question"],
                              "answerable": case["answerable"], "config": args.config, **cfg,
                              "model": client.model, **r}
        print(f"{case['id']:4s} {r['status']:8s} searches={r['n_searches']} cites={r['citations']} "
              f"{(r.get('answer') or r.get('error') or '')[:110]!r}", flush=True)
        path.write_text(json.dumps(list(traces.values()), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
