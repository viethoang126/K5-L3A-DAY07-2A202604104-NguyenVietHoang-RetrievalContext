"""Context selection: deduplicate, keep relevance order, optional neighbour chunks,
pack into a character budget (headers count). Chunks that do not fit are skipped,
not truncated, so no evidence sentence is cut mid-way."""
from typing import Any


def header(c: dict[str, Any]) -> str:
    m = c.get("metadata", {})
    pages = f"{c['page']}" if m.get("page_end", c["page"]) == c["page"] else f"{c['page']}-{m['page_end']}"
    dieu = f", Điều {m['dieu']}" if m.get("dieu", -1) not in (-1, None) else ""
    return f"[{c['id']}] (Chương {m.get('chuong')}{dieu}, trang {pages})"


def block(c: dict[str, Any]) -> str:
    return f"{header(c)}\n{c['text'].strip()}"


def with_neighbours(results: list[dict[str, Any]], by_order: dict[int, dict[str, Any]],
                    same_article_only: bool = True) -> list[dict[str, Any]]:
    """Add the next chunk of the same article after each hit (articles split into s00, s01...)."""
    out, seen = [], set()
    for r in results:
        for c in (r, by_order.get(r["metadata"]["order"] + 1)):
            if c is None or c["id"] in seen:
                continue
            if c is not r and same_article_only and c["metadata"].get("dieu") != r["metadata"].get("dieu"):
                continue
            seen.add(c["id"])
            out.append(c if c is r else {**c, "neighbour_of": r["id"]})
    return out


def pack(results: list[dict[str, Any]], budget: int) -> tuple[str, list[str]]:
    blocks, ids, used = [], [], 0
    for c in results:
        if c["id"] in ids:
            continue
        b = block(c)
        extra = len(b) + (2 if blocks else 0)
        if used + extra <= budget:
            blocks.append(b)
            ids.append(c["id"])
            used += extra
    return "\n\n".join(blocks), ids
