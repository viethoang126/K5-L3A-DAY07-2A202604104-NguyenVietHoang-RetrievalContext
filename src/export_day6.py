"""Export the Day 6 Chroma collection to data/corpus/chunks.json (Day 7 chunk schema).

Only reads the vector store; vectors stay in Chroma. Run: python -m src.export_day6
"""
import json

import chromadb

from .config import CHROMA_COLLECTION, CHROMA_PATH, CHUNKS_PATH, EMBEDDING_DIM, EMBEDDING_MODEL


def main() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_collection(CHROMA_COLLECTION)
    meta = col.metadata or {}
    if meta.get("embedding_model") != EMBEDDING_MODEL or int(meta.get("dimensions", -1)) != EMBEDDING_DIM:
        raise RuntimeError(f"Collection metadata {meta} does not match {EMBEDDING_MODEL}/{EMBEDDING_DIM}")
    data = col.get(include=["documents", "metadatas"])
    chunks = []
    for cid, text, m in zip(data["ids"], data["documents"], data["metadatas"]):
        chunks.append({"id": cid, "text": text, "source_id": m["document_id"],
                       "source": "Bộ luật Lao động 2019 (45/2019/QH14), Công báo PDF",
                       "page": int(m["page"]), "metadata": dict(m)})
    chunks.sort(key=lambda c: (c["metadata"]["char_start"], c["id"]))
    for i, c in enumerate(chunks):                       # neighbour order for context expansion
        c["metadata"]["order"] = i
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"exported {len(chunks)} chunks from {CHROMA_COLLECTION} -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
