# Record serial provider calls without automatic transport retries.

from __future__ import annotations
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import ssl
import threading
import time
import uuid
from urllib.parse import urlsplit
from ...llm import parse_json

OPENROUTER_URL = "https://openrouter.ai/api/v1"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


# Report transport failures without exposing credentials or private provider error bodies.
class ProviderError(RuntimeError):
    pass


# Serialize content consistently for request and receipt identities.
def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


# Hash structured content independently of JSON indentation.
def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


# Write private receipts exclusively and replace internal snapshots atomically.
def write_record(path: Path, value: object, *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ProviderError("Refusing a redirected receipt path")
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    target = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex) if replace else path
    try:
        with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(target, path)
    finally:
        if replace and target.exists():
            target.unlink()


# Require finite nonnegative measurements without accepting booleans as amounts.
def amount(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ProviderError(f"Missing or invalid {name}")
    return float(value)


# Use the configured certificate file or the installed CA bundle for verified TLS.
def tls_context() -> ssl.SSLContext:
    certificate = os.environ.get("SSL_CERT_FILE")
    if not certificate:
        try:
            import certifi
            certificate = certifi.where()
        except ImportError:
            certificate = None
    return ssl.create_default_context(cafile=certificate)


# Restrict each transport to its recorded provider and one request per dispatch.
class Transport:
    # Keep the selected provider credential only in memory.
    def __init__(self, token: str, *, base_url: str = OPENROUTER_URL, timeout: float = 120):
        if (not isinstance(token, str) or not token.strip() or token != token.strip()
                or token.lower().startswith(("your-", "your_", "replace-", "replace_", "<"))):
            raise ProviderError("Set the required provider key before paid execution")
        if base_url not in {OPENROUTER_URL, GOOGLE_URL}:
            raise ProviderError("The replication provider must match the fixed batch definition")
        self._token, self.base_url, self.timeout = token, base_url, timeout
        self.cost_required = base_url == OPENROUTER_URL

    # Submit a completion once without redirects, SDK retries, or error-body logging.
    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        if method != "POST" or path != "/api/v1/chat/completions" or not isinstance(body, dict):
            raise ProviderError("Only the fixed completion endpoint is supported")
        address = urlsplit(self.base_url)
        endpoint = address.path.rstrip("/") + "/chat/completions"
        connection = http.client.HTTPSConnection(address.hostname, timeout=self.timeout, context=tls_context())
        try:
            connection.request("POST", endpoint, canonical(body).encode("utf-8"),
                               {"Authorization": "Bearer " + self._token, "Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ProviderError(f"Provider HTTP {response.status}; no automatic retry")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ProviderError("Provider response exceeds the size bound")
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeError):
                raise ProviderError("Provider response was not valid JSON") from None
            if not isinstance(result, dict) or "error" in result:
                raise ProviderError("Provider returned an error or non-object response")
            return result
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Provider transport failed ({type(exc).__name__}); delivery may be uncertain") from None
        finally:
            connection.close()

    # Preserve the historical AOS wire body with no sampling, reasoning, or output overrides.
    def complete(self, model: str, prompt: str, system: str) -> dict:
        return self.request("POST", "/api/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        })


# Accumulate response-attributed charges without claiming to enforce a provider spending limit.
class Budget:
    # Bind one stage's serial transport and locally recorded charge total.
    def __init__(self, transport):
        self.transport = transport
        self.response_cost_usd = 0.0

    # Return an explicitly local meter without making account or model requests.
    def snapshot(self) -> dict:
        return {"scope": "stage_response_costs", "response_cost_usd": round(self.response_cost_usd, 12),
                "spending_limit_enforced": False}

    # Admit the next explicitly requested stage call without changing its provider settings.
    def admit(self) -> dict:
        return self.snapshot()

    # Add a known provider charge without estimating missing prices.
    def record_cost(self, cost: float) -> None:
        self.response_cost_usd = round(self.response_cost_usd + amount(cost, "response cost"), 12)


# Supply the existing AOS complete_json interface with per-call provenance.
class MeteredModel:
    # Bind the exact actor model and maximum call count for one generation stage.
    def __init__(self, budget, directory: Path, model: str, *, max_calls: int, progress=print):
        self.budget, self.directory, self.model = budget, Path(directory), model
        self.max_calls, self.progress = max_calls, progress
        self.lock, self.halted = threading.Lock(), False
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                      "cost_usd": 0.0, "unpriced_calls": 0}

    # Preserve one attempted call and parse the unchanged response without a retry.
    def complete_json(self, prompt: str, system: str | None = None) -> dict:
        with self.lock:
            if self.halted or self.usage["calls"] >= self.max_calls:
                raise ProviderError("Actor transport is stopped or its call bound is exhausted")
            if not isinstance(prompt, str) or not isinstance(system, str) or len(prompt.encode()) > 1_000_000:
                raise ProviderError("Invalid or oversized actor prompt")
            number = self.usage["calls"] + 1
            path = self.directory / "calls" / f"{number:03d}.json"
            request = {"model": self.model,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                       "response_format": {"type": "json_object"}}
            record = {"call": number, "status": "started", "model": self.model, "system": system,
                      "prompt": prompt, "max_tokens": None, "temperature": None, "request": request,
                      "base_url": getattr(self.budget.transport, "base_url", OPENROUTER_URL),
                      "meter_before": self.budget.admit(), "provider_cost_usd": None,
                      "inference_retry_policy": "disabled"}
            write_record(path, record)
            self.usage["calls"] += 1
            self.progress(f"[actor call {number}/{self.max_calls}] Sending {self.model}")
            started = time.monotonic()
            try:
                raw = self.budget.transport.request("POST", "/api/v1/chat/completions", request)
                record["response"] = raw
                usage = raw.get("usage") or {}
                if usage.get("is_byok") is True:
                    raise ProviderError("External BYOK response costs are unsupported")
                cost = usage.get("cost")
                if cost is not None or getattr(self.budget.transport, "cost_required", True):
                    cost = amount(cost, "provider response cost")
                    record["provider_cost_usd"] = cost
                    self.budget.record_cost(cost)
                    self.usage["cost_usd"] = round(self.usage["cost_usd"] + cost, 12)
                else:
                    record["cost_status"] = "not_reported_by_provider"
                for source, target in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
                    value = usage.get(source, 0)
                    if type(value) is not int or value < 0:
                        raise ProviderError("Invalid provider token count")
                    self.usage[target] += value
                reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                if type(reasoning) is not int or reasoning < 0:
                    raise ProviderError("Invalid provider reasoning token count")
                self.usage["reasoning_tokens"] += reasoning
                choice = raw["choices"][0]
                if choice.get("finish_reason") not in {"stop", "end_turn"}:
                    raise ProviderError("Completion did not finish normally; output retained")
                content = choice["message"].get("content")
                if not isinstance(content, str):
                    raise ProviderError("Completion contained no JSON text")
                value = parse_json(content)
                if not isinstance(value, dict):
                    raise ProviderError("Completion JSON is not an object")
                record.update(status="ok", meter_after=self.budget.snapshot())
                label = f"${cost:.6f}" if cost is not None else "not reported by provider"
                self.progress(f"[actor call {number}/{self.max_calls}] Completed; charge={label}; "
                              f"{time.monotonic() - started:.1f}s")
                return value
            except BaseException as exc:
                self.halted = True
                record.update(status="failed", error_type=type(exc).__name__)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ProviderError(f"Actor call {number} failed ({type(exc).__name__}); retained without retry") from None
            finally:
                if record["provider_cost_usd"] is None:
                    self.usage["unpriced_calls"] += 1
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                write_record(path, record, replace=True)
