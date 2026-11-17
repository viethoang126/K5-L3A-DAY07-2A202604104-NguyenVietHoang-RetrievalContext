"""Hit definitions shared by retrieval and context metrics.

A chunk matches an evidence piece when: same source_id, the chunk's page range
[page, page_end] covers one of the piece's pages, and the normalized evidence quote
(NFC + collapsed whitespace) is fully contained in the chunk text (Day 6 definition).

`gold` is a list of ALTERNATIVES. Each alternative is either a single piece
{source_id, pages, evidence} or a multi-evidence group {source_id, all_of: [{pages, evidence}, ...]}
where every piece is required (e.g. q11: rule in one chunk, exception in the next).

A set of chunks (top-k or final context) COVERS the gold when, for some alternative,
every required piece is matched by some chunk in the set. Hit@k = top-k covers gold;
Context Evidence Rate = final context covers gold. A multi-evidence question therefore
cannot reach Hit@1; this is reported, not hidden.
"""
import unicodedata
from typing import Any


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def _pieces(alt: dict[str, Any]) -> list[dict[str, Any]]:
    if "all_of" in alt:
        return [{"source_id": alt["source_id"], **p} for p in alt["all_of"]]
    return [alt]


def piece_hit(chunk: dict[str, Any], piece: dict[str, Any]) -> bool:
    start = int(chunk["page"])
    end = int(chunk.get("metadata", {}).get("page_end", start))
    pages = piece["pages"] if isinstance(piece.get("pages"), list) else [piece.get("page")]
    return (chunk["source_id"] == piece["source_id"] and any(start <= int(p) <= end for p in pages)
            and norm(piece["evidence"]) in norm(chunk["text"]))


def gold_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """True if the chunk contains at least one required evidence piece (rank marker)."""
    return any(piece_hit(chunk, p) for alt in gold for p in _pieces(alt))


def covers(chunks: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    return any(all(any(piece_hit(c, p) for c in chunks) for p in _pieces(alt)) for alt in gold)


def first_cover_rank(results: list[dict[str, Any]], gold: list[dict[str, Any]]) -> int | None:
    for k in range(1, len(results) + 1):
        if covers(results[:k], gold):
            return k
    return None


def frac(n: int, d: int) -> dict[str, Any]:
    return {"n": n, "N": d, "text": f"{n}/{d}", "pct": round(100 * n / d, 1) if d else None}
