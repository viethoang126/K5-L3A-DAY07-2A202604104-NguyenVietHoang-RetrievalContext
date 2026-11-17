"""Paths and settings shared by all Day 7 scripts. Override with environment variables / .env."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Existing environment variables win.
    Looks in the repo root, then its parent folder (a .env kept outside the git repo)."""
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Day 6 vector store (copied into this repo so Day 6 files are never modified).
CHROMA_PATH = Path(os.getenv("DAY6_CHROMA_PATH", ROOT / "vectorstore" / "chroma_day6"))
CHROMA_COLLECTION = os.getenv("DAY6_COLLECTION", "day06_blld_B")  # Day 6 chosen index (by_article)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = 384
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

CHUNKS_PATH = ROOT / "data" / "corpus" / "chunks.json"
QUESTIONS_PATH = ROOT / "data" / "benchmark" / "questions.json"
CASES_PATH = ROOT / "data" / "benchmark" / "cases.json"
RESULTS_DIR = ROOT / "results"

# Agent model: Gemini through its OpenAI-compatible endpoint.
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Free tier is rate limited per minute; space calls out (seconds).
GEMINI_MIN_INTERVAL = float(os.getenv("GEMINI_MIN_INTERVAL", "6.5"))
