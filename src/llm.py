"""Gemini chat client via the OpenAI-compatible endpoint (stdlib only, tool calling).

Reads GEMINI_API_KEY from the environment / .env. No mock fallback: HTTP errors that
survive retries are raised and recorded as status "error" by the agent.
"""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .config import GEMINI_BASE_URL, GEMINI_MIN_INTERVAL, GEMINI_MODEL

# Reference list prices (USD / 1M tokens) used only to ESTIMATE cost if the paid tier were
# used; free-tier runs actually cost 0. Update if the price list changes.
PRICES = {"gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40),
          "gemini-2.0-flash": (0.10, 0.40)}


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """Daily free-tier quota hit: retrying is pointless, the run must stop and resume later."""


class GeminiClient:
    def __init__(self, model: str | None = None, temperature: float = 0.0, max_retries: int = 6,
                 min_interval: float = GEMINI_MIN_INTERVAL, extra: dict[str, Any] | None = None):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY chưa được đặt (thêm vào .env, xem .env.example)")
        self.model = model or GEMINI_MODEL
        self.temperature, self.max_retries, self.min_interval = temperature, max_retries, min_interval
        self.extra = extra or {}
        self._last = 0.0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, **self.extra}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(f"{GEMINI_BASE_URL}/chat/completions", data=data, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.api_key}"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                if isinstance(out, list):     # some error payloads come back as a list
                    out = out[0]
                out["_latency_s"] = round(time.perf_counter() - t0, 3)
                out["_attempts"] = attempt + 1
                return out
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:800]
                if e.code == 429 and ("per day" in detail.lower() or "perday" in detail.lower()):
                    raise QuotaExhausted(f"HTTP 429 daily quota: {detail}") from e
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise LLMError(f"HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise LLMError(f"network error: {e}") from e
        raise LLMError("unreachable")

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        pin, pout = PRICES.get(self.model, PRICES["gemini-2.5-flash"])
        return (prompt_tokens * pin + completion_tokens * pout) / 1e6
