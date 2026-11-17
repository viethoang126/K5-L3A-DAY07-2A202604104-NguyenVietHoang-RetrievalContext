"""Retrieval methods over the Day 6 corpus. All share one interface:

    search(query, filters=None, top_k=3) -> list[dict]

Each result: id, text, source_id, source, page, metadata, score, rank, strategy.
Filters are an exact-match dict on chunk metadata (only `chuong` is used) and must come
from the user / case input, never from gold labels.

Methods
- DenseRetriever: Day 6 Chroma collection (e5-small, cosine HNSW), query prefix "query: ".
- BM25Retriever: Okapi BM25 over Vietnamese syllables + syllable bigrams (approximates
  word segmentation, because most Vietnamese words are 2 syllables).
- HybridRRF: Reciprocal Rank Fusion of dense and BM25 candidate lists (rank-based, so no
  score normalisation between cosine and BM25 is needed).
- RerankRetriever: any base retriever's candidates re-scored with a multilingual cross-encoder.
"""
import math
import re
import unicodedata
from collections import Counter
from typing import Any

from .config import CHROMA_COLLECTION, CHROMA_PATH, EMBEDDING_DIM, EMBEDDING_MODEL, RERANK_MODEL

Filters = dict[str, Any] | None


def _matches(chunk: dict[str, Any], filters: Filters) -> bool:
    return not filters or all(chunk["metadata"].get(k) == v for k, v in filters.items())


def _result(chunk: dict[str, Any], score: float, rank: int, strategy: str) -> dict[str, Any]:
    return {**chunk, "score": float(score), "rank": rank, "strategy": strategy}


# ---------------------------------------------------------------- dense (Day 6)
class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[dict[str, Any]]):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.by_id = {c["id"]: c for c in chunks}
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.col = client.get_collection(CHROMA_COLLECTION)
        meta = self.col.metadata or {}
        if meta.get("embedding_model") != EMBEDDING_MODEL or int(meta.get("dimensions", -1)) != EMBEDDING_DIM:
            raise RuntimeError(f"index built with {meta}, query model is {EMBEDDING_MODEL}/{EMBEDDING_DIM}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.prefix = "query: " if "e5" in EMBEDDING_MODEL.lower() else ""

    def search(self, query: str, filters: Filters = None, top_k: int = 3) -> list[dict[str, Any]]:
        vec = self.model.encode(self.prefix + query, normalize_embeddings=True, show_progress_bar=False).tolist()
        if len(vec) != EMBEDDING_DIM:
            raise RuntimeError("query embedding dimension mismatch")
        where = dict(filters) if filters else None
        res = self.col.query(query_embeddings=[vec], n_results=top_k, where=where,
                             include=["distances"])
        out = []
        for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), 1):
            out.append(_result(self.by_id[cid], 1.0 - dist, rank, self.name))  # cosine similarity
        return out


# ---------------------------------------------------------------- lexical
def tokenize(text: str, bigrams: bool = True) -> list[str]:
    syl = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])] if bigrams else syl


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75, bigrams: bool = True):
        self.chunks, self.k1, self.b, self.bigrams = chunks, k1, b, bigrams
        self.tf = [Counter(tokenize(c["text"], bigrams)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, filters: Filters = None, top_k: int = 3) -> list[dict[str, Any]]:
        q = Counter(tokenize(query, self.bigrams))
        scored = []
        for i, c in enumerate(self.chunks):
            if not _matches(c, filters):
                continue
            tf, s = self.tf[i], 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [_result(c, s, r, self.name) for r, (s, c) in enumerate(scored[:top_k], 1)]


# ---------------------------------------------------------------- hybrid
class HybridRRF:
    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, lexical: BM25Retriever, candidates: int = 20, k: int = 60):
        self.dense, self.lexical, self.candidates, self.k = dense, lexical, candidates, k

    def search(self, query: str, filters: Filters = None, top_k: int = 3) -> list[dict[str, Any]]:
        fused: dict[str, float] = {}
        chunk_by_id: dict[str, dict[str, Any]] = {}
        for retr in (self.dense, self.lexical):
            for r in retr.search(query, filters, self.candidates):
                fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (self.k + r["rank"])
                chunk_by_id[r["id"]] = {k: v for k, v in r.items() if k not in ("score", "rank", "strategy")}
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        return [_result(chunk_by_id[i], fused[i], r, self.name) for r, i in enumerate(order, 1)]


# ---------------------------------------------------------------- rerank
def load_cross_encoder(model_name: str = RERANK_MODEL):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu", max_length=512)


class RerankRetriever:
    def __init__(self, base, candidates: int = 20, model=None):
        self.base, self.candidates = base, candidates
        self.model = model if model is not None else load_cross_encoder()
        self.name = f"{base.name}+rerank"

    def search(self, query: str, filters: Filters = None, top_k: int = 3) -> list[dict[str, Any]]:
        cands = self.base.search(query, filters, self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands], show_progress_bar=False)
        ranked = sorted(zip(scores, cands), key=lambda x: (-float(x[0]), x[1]["id"]))[:top_k]
        return [_result({k: v for k, v in c.items() if k not in ("score", "rank", "strategy")}, s, r, self.name)
                | {"base_rank": c["rank"]} for r, (s, c) in enumerate(ranked, 1)]


def build(method: str, chunks: list[dict[str, Any]], *, candidates: int = 20, cache: dict | None = None):
    """Factory with a shared cache so heavy models load once per process."""
    cache = cache if cache is not None else {}

    def get(key, make):
        if key not in cache:
            cache[key] = make()
        return cache[key]

    dense = lambda: get("dense", lambda: DenseRetriever(chunks))
    bm25 = lambda: get("bm25", lambda: BM25Retriever(chunks))
    if method == "dense":
        return dense()
    if method == "bm25":
        return bm25()
    if method == "bm25_unigram":
        return get("bm25u", lambda: BM25Retriever(chunks, bigrams=False))
    if method == "hybrid":
        return HybridRRF(dense(), bm25(), candidates=candidates)
    ce = lambda: get("ce", load_cross_encoder)
    if method == "dense_rerank":
        return RerankRetriever(dense(), candidates, ce())
    if method == "hybrid_rerank":
        return RerankRetriever(HybridRRF(dense(), bm25(), candidates=candidates), candidates, ce())
    raise ValueError(f"unknown method {method}")
