"""Thin client for the Gemini models served through the lab's OpenAI-compatible proxy.

Configuration comes from ``AOS/.env``: ``API_KEY``, ``BASE_URL``, optionally
``AOS_MODEL`` (default ``gemini-3.6-flash``) and ``AOS_PRICES`` (JSON mapping a
model to its [input, output] price in USD per million tokens). Every call's token
usage is accumulated so steps and tasks can be costed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

AOS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-3.6-flash"

#: USD per million tokens (input, output). The proxy publishes no prices, so these are
#: list-price assumptions for the Flash and Pro classes; override with AOS_PRICES.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-pro-preview": (1.25, 10.00),
}

#: ``output_tokens`` includes thinking (reasoning) tokens, which the proxy counts inside
#: ``completion_tokens`` and Gemini bills at the output rate; ``reasoning_tokens`` is their share.
ZERO_USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0,
              "unpriced_calls": 0}


def load_env() -> None:
    """Load ``AOS/.env`` (and a ``.env`` in the current directory) without overriding the environment."""
    load_dotenv(AOS_ROOT / ".env", override=False)
    load_dotenv(override=False)


def load_prices() -> dict[str, tuple[float, float]]:
    prices = dict(DEFAULT_PRICES)
    raw = os.environ.get("AOS_PRICES", "").strip()
    if raw:
        try:
            for model, pair in json.loads(raw).items():
                prices[model] = (float(pair[0]), float(pair[1]))
        except (ValueError, TypeError, IndexError, AttributeError) as exc:
            raise RuntimeError(f"AOS_PRICES must be JSON like {{\"model\": [input, output]}} in USD per 1M tokens: {exc}") from None
    return prices


def price_usd(model: str | None, input_tokens: int, output_tokens: int,
              prices: dict[str, tuple[float, float]] | None = None) -> float | None:
    """Cost of a call, or None when the model has no known price."""
    table = prices if prices is not None else load_prices()
    pair = table.get(model or "")
    if pair is None:
        return None
    return round(input_tokens / 1e6 * pair[0] + output_tokens / 1e6 * pair[1], 6)


def add_usage(*parts: dict | None) -> dict:
    """Sum usage records; ``cost_usd`` stays a number and unpriced calls are counted."""
    total = dict(ZERO_USAGE)
    for part in parts:
        if not part:
            continue
        for key in ("calls", "input_tokens", "output_tokens", "reasoning_tokens", "unpriced_calls"):
            total[key] += int(part.get(key) or 0)
        total["cost_usd"] = round(total["cost_usd"] + float(part.get("cost_usd") or 0.0), 6)
    return total


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating code fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


class LLM:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        load_env()
        self.model = model or os.environ.get("AOS_MODEL", DEFAULT_MODEL)
        api_key = api_key or os.environ.get("API_KEY")
        base_url = base_url or os.environ.get("BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("API_KEY and BASE_URL must be set in AOS/.env (see .env.example)")
        if api_key.startswith("your-") or base_url.startswith("your-"):
            raise RuntimeError("AOS/.env still has the placeholder values from .env.example; put the real API_KEY in it")
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3, timeout=600)
        self.prices = load_prices()
        self.usage = dict(ZERO_USAGE)
        self._models: list[str] | None = None

    # ----------------------------------------------------------------- calls
    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             json_mode: bool = False, temperature: float | None = None):
        """One chat completion; returns the assistant message object.

        No ``max_tokens`` is set on purpose: reasoning tokens count against it and a
        low cap returns an empty reply.
        """
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None) or 0
            self.record(usage.prompt_tokens or 0, usage.completion_tokens or 0, reasoning)
        return response.choices[0].message

    def complete(self, prompt: str, system: str | None = None) -> str:
        return self.chat(_messages(prompt, system)).content or ""

    def complete_json(self, prompt: str, system: str | None = None) -> dict:
        return parse_json(self.chat(_messages(prompt, system), json_mode=True).content or "")

    # ----------------------------------------------------------------- usage
    def record(self, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> None:
        """Add one call's tokens to the running total, priced for the current model.

        ``output_tokens`` must already include the thinking tokens (as the API reports them);
        ``reasoning_tokens`` only records how many of them were thinking.
        """
        cost = price_usd(self.model, input_tokens, output_tokens, self.prices)
        self.usage["calls"] += 1
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens
        self.usage["reasoning_tokens"] += reasoning_tokens
        if cost is None:
            self.usage["unpriced_calls"] += 1
        else:
            self.usage["cost_usd"] = round(self.usage["cost_usd"] + cost, 6)

    def snapshot(self) -> dict:
        return dict(self.usage)

    def usage_since(self, snapshot: dict | None) -> dict:
        if not snapshot:
            return dict(self.usage)
        out = {k: self.usage[k] - snapshot.get(k, 0) for k in ZERO_USAGE}
        out["cost_usd"] = round(out["cost_usd"], 6)
        return out

    # ---------------------------------------------------------------- models
    def available_models(self) -> list[str]:
        """Models the key can use, as the proxy lists them (falls back to the current model)."""
        if self._models is None:
            try:
                self._models = sorted(m.id for m in self.client.models.list())
            except Exception:  # listing is a convenience; the configured model still works
                self._models = []
        return self._models or [self.model]


def _messages(prompt: str, system: str | None) -> list[dict]:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return messages
