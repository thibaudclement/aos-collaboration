# Run the frozen PiSAs actors and judges through one metered, non-retrying transport.

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

from .provider import amount, digest, write_record

UPSTREAM_REVISION = "1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd"
ACTOR_MODEL = "google/gemini-3.5-flash-lite"
OPUS_MODEL = "anthropic/claude-opus-5.5"
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "pisas"
ACTOR_TEMPERATURE = 0.7
MAX_ROUNDS = 30
MAX_OUTPUT_TOKENS = 16384
MAX_NATIVE_CALLS = MAX_ROUNDS + 2
JUDGE_GAP_SECONDS = 2
JUDGES = ("google/gemini-2.5-pro", "anthropic/claude-haiku-4.5",
          "openai/gpt-4o-mini", "google/gemini-2.5-flash")
JUDGE_ARGS = ["--judge-mode", "chain", "--judge-llm", JUDGES[0],
              "--verifier-llm-1", JUDGES[1], "--verifier-llm-2", JUDGES[2],
              "--verifier-llm-3", JUDGES[3], "--verifier-max-tokens", "1024",
              "--judge-workers", "1", "--agent-audit"]
SOURCE_SHA256 = {
    "pipeline.py": "ef2f76c164339915f9c330a4da384fe8c701144511cd289c39150d3fe0f47237",
    "benchmark.py": "6ac8b77a5d7645917a0661ae4d9451f39aa96881ab4281c15794820aafb07d3d",
    "metrics.py": "4eb7adfa06a2bfe88035eba0f6b414eb935b7ae64d73d55fa3ef53d7c86963b8",
    "agents.py": "4ab8ae176ab3dbefdaf0519e285f913d2e00a081df734ffc9f707916c52c0d81",
    "judges.py": "9a437860504e756b39366552753507058db709ecd760d7309a0c9b97f1984d37",
    "fact_judge.py": "3e43568a078ba0ccf8bc440370994e98c23183c4d6d1adaa8f2785ca9962746c",
    "run_pipeline.py": "bcc45974f895b9364c1f2098a9c8a3b86be6a47e2c8814bb8ba8f624060699ab",
    "run_evaluation.py": "768aed92e501ed4be990b009c00a822eaedb0c244aa437ac320ecaee453b6747"
}
_MODULES = ("pipeline", "benchmark", "metrics", "agents", "judges", "fact_judge",
            "run_pipeline", "run_evaluation")
_UPSTREAM_LOCK = threading.RLock()
_PLACEHOLDER_KEY = "parent-budget-transport-only"


# Report a stage failure without exposing provider text or discarding incurred charges.
class NativeError(RuntimeError):
    # Keep usage and receipt identities available to the enclosing stage recorder.
    def __init__(self, message, *, usage=None, receipts=None):
        super().__init__(message)
        self.usage = deepcopy(usage or {})
        self.receipts = deepcopy(receipts or [])


# Escape native retry and verifier except-Exception fallbacks without another inference.
class InferenceHalt(BaseException):
    pass


# Check only frozen evaluator source bytes, without opening any scenario contents.
def source_fingerprint(upstream: Path = VENDOR_ROOT) -> dict:
    upstream = Path(upstream).expanduser().resolve(strict=True)
    actual = {str(p.relative_to(upstream)) for p in upstream.rglob("*")
              if p.is_file() and p.suffix in (".py", ".txt")}
    if actual != set(SOURCE_SHA256):
        raise NativeError("The frozen PiSAs source inventory changed")
    for name, expected in SOURCE_SHA256.items():
        path = upstream / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise NativeError("The frozen PiSAs source changed: " + name)
    return {"revision": UPSTREAM_REVISION, "files": dict(SOURCE_SHA256)}


# Expose the exact legacy behavior and the separately declared transport boundary.
def specification(actor_model: str = ACTOR_MODEL) -> dict:
    return {"upstream_revision": UPSTREAM_REVISION, "actor_model": actor_model,
            "actor_temperature": None if actor_model == OPUS_MODEL else ACTOR_TEMPERATURE, "native_system": "decentralized",
            "privacy_level": "High", "private_memory": False, "shared_memory": False,
            "max_rounds": MAX_ROUNDS, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "hide_task_from_peers": False, "maximum_native_calls": MAX_NATIVE_CALLS,
            "judge_args": list(JUDGE_ARGS), "judge_temperature": 0.0,
            "judge_completion_gap_seconds": JUDGE_GAP_SECONDS,
            "judge_output_audit": True, "judge_output_leak": True,
            "judge_agent_audit": True, "judge_memory_audit": False,
            "judge_roster": True, "judge_violation_leniency": True,
            "transport": "shared parent Budget; serial receipts; no inference retries; unchanged provider routing",
            "runtime": "frozen upstream Python source through a transport-only completion shim"}


# Preserve native message parameters while metering every dispatch under the existing account.
class MeteredCompletion:
    # Serialize nested verifier threads and bind the stage's allowed models and call ceiling.
    def __init__(self, budget, directory: Path, *, kind: str, max_calls: int, progress=print,
                 sleep=time.sleep, clock=time.monotonic, gap=None, actor_model=ACTOR_MODEL):
        if kind not in {"native", "judge"} or type(max_calls) is not int or not 1 <= max_calls <= 10000:
            raise NativeError("Choose a known completion kind and a bounded positive call count")
        if kind == "native" and actor_model not in {ACTOR_MODEL, OPUS_MODEL}:
            raise NativeError("Unknown fixed actor model")
        self.actor_model = actor_model
        self.budget, self.directory = budget, Path(directory)
        self.kind, self.max_calls, self.progress = kind, max_calls, progress
        self.lock, self.halted = threading.Lock(), False
        self.sleep, self.clock = sleep, clock
        self.gap = (JUDGE_GAP_SECONDS if kind == "judge" else 0) if gap is None else gap
        self.ready_at = 0.0
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "reasoning_tokens": 0, "cost_usd": 0.0, "unpriced_calls": 0}

    # Accept only the frozen upstream OpenRouter parameters and exclude all credentials from receipts.
    def _body(self, args, kwargs) -> dict:
        expected = {"messages", "max_tokens", "model", "api_key", "api_base", "temperature",
                    "num_retries", "timeout"}
        if args or set(kwargs) != expected or kwargs["api_key"] != _PLACEHOLDER_KEY:
            raise NativeError("Native completion parameters differ from the frozen route")
        if kwargs["api_base"] != "https://openrouter.ai/api/v1" or kwargs["timeout"] != 120:
            raise NativeError("Native completion route or timeout changed")
        model = kwargs["model"]
        allowed = (self.actor_model,) if self.kind == "native" else JUDGES
        if model not in {"openrouter/" + item for item in allowed}:
            raise NativeError("Unexpected native or judge model")
        model = model.removeprefix("openrouter/")
        temperature = ACTOR_TEMPERATURE if self.kind == "native" else 0.0
        tokens = kwargs["max_tokens"]
        valid_tokens = {MAX_OUTPUT_TOKENS} if self.kind == "native" else (
            {4096, 16384} if model == JUDGES[0] else {1024})
        if kwargs["temperature"] != temperature or type(tokens) is not int or tokens not in valid_tokens:
            raise NativeError("Native completion temperature or token budget changed")
        messages = kwargs["messages"]
        if (not isinstance(messages, list) or len(messages) != 1
                or set(messages[0]) != {"role", "content"} or messages[0]["role"] != "user"
                or not isinstance(messages[0]["content"], str)
                or len(messages[0]["content"].encode("utf-8")) > 1_000_000):
            raise NativeError("Unexpected native message shape or size")
        body = {"model": model, "messages": deepcopy(messages), "max_tokens": tokens,
                "temperature": temperature}
        if self.kind == "native" and model == OPUS_MODEL:
            del body["temperature"]
        return body

    # Keep the frozen parser's acceptance rule while stopping its otherwise repeated malformed inference.
    def _response(self, raw, body):
        choice = raw["choices"][0]
        if choice.get("finish_reason") not in {"stop", "end_turn"}:
            raise NativeError("Native completion did not finish normally")
        message = choice["message"]
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise NativeError("Native completion returned a nontext message")
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usable = content if content.strip() else (reasoning if self.kind == "native" else "")
        if not isinstance(usable, str) or not usable.strip():
            raise NativeError("Native completion was empty; no inference retry")
        if self.kind == "judge" and body["model"] == JUDGES[0]:
            # This is exactly judges._parse_json's slice, with its retry converted into a retained failure.
            try:
                parsed = json.loads(content[content.find("{"):content.rfind("}") + 1])
            except (ValueError, TypeError):
                parsed = {}
            if not parsed:
                raise NativeError("Judge JSON would require another inference; retained without retry")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(**message))])

    # Retain one paid attempt before dispatch and halt all queued calls on any uncertain or malformed result.
    def __call__(self, *args, **kwargs):
        with self.lock:
            if self.halted:
                raise InferenceHalt("The stage transport is already halted")
            path = record = None
            started = time.monotonic()
            try:
                if self.usage["calls"] >= self.max_calls:
                    raise NativeError("Stage call bound reached; no inference started")
                body = self._body(args, kwargs)
                delay = self.ready_at - self.clock()
                if delay > 0:
                    self.sleep(delay)
                before = self.budget.admit()
                number = self.usage["calls"] + 1
                path = self.directory / "calls" / f"{number:03d}.json"
                record = {"call": number, "status": "started", "request": body,
                          "meter_before": before, "provider_cost_usd": None,
                          "inference_retry_policy": "disabled"}
                if self.kind == "native" and self.actor_model == OPUS_MODEL:
                    record["actor_model_compatibility"] = compatibility_record(body)
                write_record(path, record)
                self.usage["calls"] += 1
                self.progress(f"[call {number}/{self.max_calls}] Sending one bounded {self.kind} completion")
                raw = self.budget.transport.request("POST", "/api/v1/chat/completions", body)
                self.ready_at = self.clock() + self.gap
                record["response"] = raw
                usage = raw.get("usage") or {}
                cost = amount(usage.get("cost"), "provider response cost")
                if usage.get("is_byok") is True:
                    raise NativeError("External BYOK costs are unsupported")
                record["provider_cost_usd"] = cost
                self.budget.record_cost(cost)
                self.usage["cost_usd"] = round(self.usage["cost_usd"] + cost, 12)
                for key, target in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
                    value = usage.get(key, 0)
                    if type(value) is not int or value < 0:
                        raise NativeError("Invalid provider token count")
                    self.usage[target] += value
                details = usage.get("completion_tokens_details") or {}
                reasoning = details.get("reasoning_tokens") or 0
                if type(reasoning) is not int or reasoning < 0:
                    raise NativeError("Invalid provider reasoning token count")
                self.usage["reasoning_tokens"] += reasoning
                response = self._response(raw, body)
                record.update(status="ok", meter_after=self.budget.snapshot(),
                              elapsed_seconds=round(time.monotonic() - started, 3))
                write_record(path, record, replace=True)
                self.progress(f"[call {number}/{self.max_calls}] Completed; response charge ${cost:.6f}")
                return response
            except BaseException as exc:
                self.halted = True
                if record is not None:
                    record.update(status="failed", error_type=type(exc).__name__,
                                  elapsed_seconds=round(time.monotonic() - started, 3))
                    if record["provider_cost_usd"] is None:
                        self.usage["unpriced_calls"] += 1
                    write_record(path, record, replace=True)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise InferenceHalt(f"Completion stopped ({type(exc).__name__}); no inference retry") from None

    # Return receipt paths and byte identities without duplicating prompt or evaluator content.
    def receipts(self) -> list[dict]:
        return [{"path": str(path.relative_to(self.directory)),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted((self.directory / "calls").glob("*.json"))]


# Isolate upstream globals and redirect only its SDK transport while retaining every source file.
@contextmanager
def _loaded(upstream: Path, completion):
    with _UPSTREAM_LOCK:
        source_fingerprint(upstream)
        if os.environ.get("REMOTE_MODEL_EXTRA_BODY"):
            raise NativeError("Unset REMOTE_MODEL_EXTRA_BODY before the frozen native comparison")
        names = ("litellm",) + _MODULES
        missing = object()
        saved = {name: sys.modules.get(name, missing) for name in names}
        old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
        loggers = {name: logging.getLogger(name) for name in ("pipeline", "evaluation")}
        logger_state = {name: (list(log.handlers), log.level, log.propagate, log.disabled)
                        for name, log in loggers.items()}
        try:
            for log in loggers.values():
                log.handlers = []
                log.propagate, log.disabled = False, False
            for name in names:
                sys.modules.pop(name, None)
            shim = ModuleType("litellm")
            shim.completion = completion
            sys.modules["litellm"] = shim
            sys.path.insert(0, str(upstream))
            sys.dont_write_bytecode = True
            modules = {}
            for name in _MODULES:
                spec = importlib.util.spec_from_file_location(name, upstream / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                modules[name] = module
            yield modules
        finally:
            sys.path[:] = old_path
            sys.dont_write_bytecode = old_bytecode
            for name, value in saved.items():
                if value is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
            for name, log in loggers.items():
                handlers, level, propagate, disabled = logger_state[name]
                for handler in list(log.handlers):
                    if handler not in handlers:
                        handler.close()
                log.handlers[:] = handlers
                log.setLevel(level)
                log.propagate, log.disabled = propagate, disabled


# Validate completed native execution separately from the meaning or correctness of its answer.
def pipeline_health(pipeline: dict, scenario_root: Path, *, actor_model=ACTOR_MODEL) -> dict:
    config = pipeline["config"]
    expected = {"scenario": scenario_root.name, "system": "decentralized", "agent_llm": actor_model,
                "privacy_level": "High", "private_memory": False, "shared_memory": False,
                "max_rounds": MAX_ROUNDS, "max_output_tokens": MAX_OUTPUT_TOKENS,
                "hide_task_from_peers": False}
    if any(config.get(key) != value for key, value in expected.items()):
        raise NativeError("Native pipeline settings changed")
    if Path(config.get("scenario_dir", "")).resolve() != scenario_root.resolve():
        raise NativeError("Native pipeline scenario path changed")
    if (pipeline.get("degraded") is not False or pipeline.get("n_empty_generations") != 0
            or not isinstance(pipeline.get("final_response"), str) or not pipeline["final_response"].strip()
            or not isinstance(pipeline.get("agent_contents"), dict) or not pipeline["agent_contents"]
            or not isinstance(pipeline.get("log"), list) or type(pipeline.get("n_a2a")) is not int):
        raise NativeError("Incomplete native pipeline; preserve the attempt")
    return {"degraded": False, "n_empty_generations": 0, "n_a2a": pipeline["n_a2a"]}


# Bind the exact effective Opus request to its nominal upstream sampling setting.
def compatibility_record(request: dict) -> dict:
    original = deepcopy(request)
    original["temperature"] = ACTOR_TEMPERATURE
    return {"profile": "aos-opus-5.5-actor-compatibility-v1", "kind": "native",
            "model": OPUS_MODEL, "upstream_temperature": ACTOR_TEMPERATURE,
            "effective_temperature": None, "temperature_omitted_as_unsupported": True,
            "upstream_request_sha256": digest(original), "effective_request_sha256": digest(request),
            "reasoning_parameter_sent": False, "thinking_policy": "model_intrinsic_adaptive",
            "documented_default_effort": "medium"}


# Run the pinned native algorithm through one metered completion adapter.
def generate(scenario_root, out, budget, *, actor_model=ACTOR_MODEL,
             max_calls=MAX_NATIVE_CALLS, progress=print) -> dict:
    if actor_model not in {ACTOR_MODEL, OPUS_MODEL} or not 1 <= max_calls <= MAX_NATIVE_CALLS:
        raise NativeError("Native model or call ceiling differs from the fixed comparison")
    scenario_root = Path(scenario_root).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve()
    with _UPSTREAM_LOCK:
        source_fingerprint()
        if out.exists():
            raise NativeError("Choose a new native stage directory; saved inference is never repeated")
        out.mkdir(parents=True, mode=0o700)
        console, errors = sys.stdout, sys.stderr

        # Keep progress visible while retaining upstream diagnostic output in its own log.
        def notice(message):
            with redirect_stdout(console), redirect_stderr(errors):
                progress(message)

        adapter = MeteredCompletion(budget, out, kind="native", max_calls=max_calls,
                                    progress=notice, actor_model=actor_model)
        stage_root = out / "upstream"
        stage_root.mkdir(mode=0o700)
        old_umask = os.umask(0o077)
        timeout = getattr(budget.transport, "timeout", 120)
        budget.transport.timeout = 120
        try:
            with (out / "upstream.log").open("x", encoding="utf-8") as transcript:
                with redirect_stdout(transcript), redirect_stderr(transcript), _loaded(VENDOR_ROOT, adapter) as modules:
                    runner = modules["run_pipeline"]
                    args = runner.build_parser().parse_args(
                        ["-d", str(scenario_root), "-o", str(stage_root), "--system", "decentralized",
                         "--agent-llm", actor_model, "--privacy-level", "High",
                         "--max-rounds", str(MAX_ROUNDS), "--max-output-tokens", str(MAX_OUTPUT_TOKENS)])
                    runner.run_single(args, _PLACEHOLDER_KEY)
            result_path = stage_root / scenario_root.name / f"pipeline_{scenario_root.name}.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            health = pipeline_health(result, scenario_root, actor_model=actor_model)
            if adapter.halted or adapter.usage["calls"] == 0:
                raise NativeError("Native stage made no successful completion")
            return {"pipeline": result, "usage": deepcopy(adapter.usage),
                    "receipts": adapter.receipts(), "health": health,
                    "upstream_output": str(result_path),
                    "upstream_output_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()}
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise NativeError(f"Native stage stopped ({type(exc).__name__}); preserve this attempt",
                              usage=adapter.usage, receipts=adapter.receipts()) from None
        finally:
            budget.transport.timeout = timeout
            os.umask(old_umask)
