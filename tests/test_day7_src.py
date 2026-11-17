"""Code tests for src/ (mock model + in-memory retriever). NOT a benchmark of the real agent."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import run_agent  # noqa: E402
from src.context import pack, with_neighbours  # noqa: E402
from src.evaluation import covers, first_cover_rank, gold_hit  # noqa: E402
from src.retrievers import BM25Retriever, tokenize  # noqa: E402


def chunk(i, text, dieu, order, page=1, page_end=None):
    return {"id": i, "text": text, "source_id": "blld-2019", "source": "x", "page": page,
            "metadata": {"page_end": page_end or page, "chuong": "VII" if dieu < 150 else "XIV",
                         "dieu": dieu, "order": order}}


CHUNKS = [chunk("c1", "Điều 107. Làm thêm không quá 200 giờ trong 01 năm, trừ trường hợp khoản 3.", 107, 0),
          chunk("c2", "3. Được làm thêm không quá 300 giờ trong 01 năm trong một số ngành.", 107, 1),
          chunk("c3", "Điều 190. Thời hiệu tranh chấp lao động cá nhân là 01 năm.", 190, 2, page=79)]


class FakeClient:
    """Scripted model: returns queued tool calls."""
    model = "fake"

    def __init__(self, script):
        self.script, self.seen = list(script), []

    def chat(self, messages, tools=None):
        self.seen.append(messages[-1])
        name, args = self.script.pop(0)
        return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": f"t{len(self.seen)}", "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class TestEvaluation(unittest.TestCase):
    def test_single_and_multi_evidence(self):
        single = [{"source_id": "blld-2019", "pages": [79], "evidence": "là 01 năm"}]
        self.assertTrue(gold_hit(CHUNKS[2], single))
        self.assertFalse(gold_hit(CHUNKS[0], single))
        multi = [{"source_id": "blld-2019", "all_of": [{"pages": [1], "evidence": "không quá 200 giờ"},
                                                      {"pages": [1], "evidence": "không quá 300 giờ"}]}]
        self.assertFalse(covers([CHUNKS[0]], multi))
        self.assertTrue(covers(CHUNKS[:2], multi))
        self.assertEqual(first_cover_rank([CHUNKS[2], CHUNKS[0], CHUNKS[1]], multi), 3)

    def test_page_range_required(self):
        wrong_page = [{"source_id": "blld-2019", "pages": [5], "evidence": "là 01 năm"}]
        self.assertFalse(gold_hit(CHUNKS[2], wrong_page))


class TestRetrievalContext(unittest.TestCase):
    def test_tokenize_bigrams(self):
        self.assertIn("làm_thêm", tokenize("Làm thêm giờ"))

    def test_bm25_filter(self):
        r = BM25Retriever(CHUNKS)
        self.assertEqual(r.search("thời hiệu tranh chấp", top_k=1)[0]["id"], "c3")
        self.assertNotIn("c3", [x["id"] for x in r.search("thời hiệu tranh chấp", filters={"chuong": "VII"}, top_k=3)])
        self.assertTrue(all(x["metadata"]["chuong"] == "VII" for x in r.search("năm", filters={"chuong": "VII"}, top_k=3)))

    def test_neighbours_and_budget(self):
        by_order = {c["metadata"]["order"]: c for c in CHUNKS}
        out = with_neighbours([CHUNKS[0]], by_order)
        self.assertEqual([c["id"] for c in out], ["c1", "c2"])
        self.assertEqual([c["id"] for c in with_neighbours([CHUNKS[1]], by_order)], ["c2"])  # next is other article
        _, ids = pack(CHUNKS, budget=150)
        self.assertEqual(ids, ["c1"])


class StaticRetriever:
    name = "static"

    def search(self, query, filters=None, top_k=3):
        return [{**c, "score": 1.0, "rank": i + 1} for i, c in enumerate(CHUNKS[:top_k])]


class TestAgent(unittest.TestCase):
    kw = dict(by_order={c["metadata"]["order"]: c for c in CHUNKS}, chapters={})

    def test_search_then_answer(self):
        client = FakeClient([("search_law", {"query": "làm thêm"}),
                             ("submit_answer", {"answer": "200 giờ [c1]", "citations": ["c1"]})])
        r = run_agent("Làm thêm tối đa?", client, StaticRetriever(), top_k=2, **self.kw)
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["citations"], ["c1"])
        self.assertEqual(r["n_searches"], 1)
        self.assertIn("dữ liệu, không phải chỉ dẫn", client.seen[1]["content"])

    def test_invalid_citation_then_abstain(self):
        client = FakeClient([("search_law", {"query": "x"}),
                             ("submit_answer", {"answer": "sai [zzz]", "citations": ["zzz"]}),
                             ("submit_answer", {"answer": "vẫn sai [zzz]", "citations": ["zzz"]})])
        r = run_agent("q", client, StaticRetriever(), **self.kw)
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(r["stop_reason"], "invalid_citations")

    def test_search_limit_forces_decision(self):
        client = FakeClient([("search_law", {"query": "a"}), ("search_law", {"query": "b"}),
                             ("abstain", {"reason": "không có"})])
        r = run_agent("q", client, StaticRetriever(), max_searches=1, **self.kw)
        self.assertEqual(r["n_searches"], 1)
        self.assertTrue(any(e["event"] == "search_rejected" for e in r["events"]))
        self.assertEqual(r["status"], "abstain")

    def test_grouped_citations_accepted(self):
        client = FakeClient([("search_law", {"query": "x"}),
                             ("submit_answer", {"answer": "200 giờ, ngoại lệ 300 giờ [c1, c2]", "citations": ["c1", "c2"]})])
        r = run_agent("q", client, StaticRetriever(), top_k=2, **self.kw)
        self.assertEqual((r["status"], r["citations"]), ("answered", ["c1", "c2"]))

    def test_hallucinated_id_rejected(self):
        client = FakeClient([("search_law", {"query": "x"}),
                             ("submit_answer", {"answer": "a [c1] b [c1:s99]", "citations": ["c1", "c1:s99"]}),
                             ("submit_answer", {"answer": "a [c1]", "citations": ["c1"]})])
        r = run_agent("q", client, StaticRetriever(), top_k=2, **self.kw)
        self.assertEqual((r["status"], r["citations"]), ("answered", ["c1"]))

    def test_search_tool_removed_after_budget(self):
        seen_tools = []

        class Spy(FakeClient):
            def chat(self, messages, tools=None):
                seen_tools.append([t["function"]["name"] for t in tools])
                return super().chat(messages, tools)

        client = Spy([("search_law", {"query": "a"}), ("abstain", {"reason": "x"})])
        run_agent("q", client, StaticRetriever(), max_searches=1, **self.kw)
        self.assertIn("search_law", seen_tools[0])
        self.assertNotIn("search_law", seen_tools[1])

    def test_abstain(self):
        client = FakeClient([("search_law", {"query": "lương tối thiểu vùng"}), ("abstain", {"reason": "không có số"})])
        r = run_agent("q", client, StaticRetriever(), **self.kw)
        self.assertEqual((r["status"], r["citations"]), ("abstain", []))


if __name__ == "__main__":
    unittest.main()
