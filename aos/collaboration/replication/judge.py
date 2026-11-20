# Run the pinned official evaluator with fixed request profiles and retained paid-attempt receipts.

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from . import native
from .provider import amount, digest, write_record

CAPACITY_PROFILE = "aos-judge-output-capacity-v1"
LEGACY_PROFILE = "aos-judge-legacy-v1"
PRIMARY_MAX_TOKENS = 16384
JUDGE_REQUEST_TIMEOUT_SECONDS = 300
JSON_ATTEMPTS = 5
EMPTY_TEXT_ATTEMPTS = 5


# Describe the fixed judge request allowances and bounded historical response loops.
def specification(profile: str = "capacity") -> dict:
    if profile not in {"legacy", "capacity"}:
        raise native.NativeError("Unknown fixed judging profile")
    return {"id": CAPACITY_PROFILE if profile == "capacity" else LEGACY_PROFILE,
            "judge_mode": "chain", "primary_model": native.JUDGES[0],
            "verifier_models": list(native.JUDGES[1:]), "judge_temperature": 0.0,
            "primary_max_tokens": PRIMARY_MAX_TOKENS if profile == "capacity" else None,
            "primary_output_policy": "uniform_16384" if profile == "capacity" else "upstream_requested",
            "verifier_max_tokens": 1024,
            "judge_http_timeout_seconds": JUDGE_REQUEST_TIMEOUT_SECONDS if profile == "capacity" else 120,
            "primary_json_parse_attempts": JSON_ATTEMPTS,
            "empty_text_attempts_per_upstream_call": EMPTY_TEXT_ATTEMPTS,
            "maximum_received_responses_per_primary_json_scope": JSON_ATTEMPTS * EMPTY_TEXT_ATTEMPTS,
            "source_of_retry_loops": "Unmodified pinned judges._call and judges._call_json.",
            "primary_response_acceptance": "Preserve the first-brace/last-brace JSON parser and truthiness rule.",
            "finish_reason_policy": "Preserve content parsing and record non-normal finish reasons.",
            "reasoning_only_judge_content": "Never substitute reasoning for empty content.",
            "exhaustion": "Exhausted primary JSON or direct-text scopes fail the stage.",
            "transport_retry": "Disabled; uncertain delivery halts queued calls.",
            "score_selection": "Accept the first parser-accepted response independently of score.",
            "prompts_changed": False, "models_changed": False}


# Apply the declared capacity amendment only to the primary judging model.
def effective_max_tokens(model: str, requested: int, profile: str = "capacity") -> int:
    return PRIMARY_MAX_TOKENS if profile == "capacity" and model == native.JUDGES[0] else requested


# Observe the frozen parser's acceptance rule without repairing or completing model output.
def primary_json_usable(content: str) -> bool:
    try:
        return bool(json.loads(content[content.find("{"):content.rfind("}") + 1]))
    except (ValueError, TypeError):
        return False


# Display only standard finish labels while retaining any original provider value in private receipts.
def finish_label(value):
    known = {None, "stop", "end_turn", "length", "content_filter", "tool_calls", "function_call",
             "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER", "MALFORMED_FUNCTION_CALL"}
    return value if isinstance(value, (str, type(None))) and value in known else "other"


# Meter received judging candidates while allowing only the frozen evaluator's response loops.
class MeteredJudgeCompletion(native.MeteredCompletion):
    # Keep the original request validation, pacing, receipt layout and account admission boundary.
    def __init__(self, budget, directory: Path, *, max_calls: int, progress=print,
                 sleep=time.sleep, clock=time.monotonic, gap=None, profile="capacity",
                 actor_model=native.ACTOR_MODEL):
        self.profile = profile
        self.profile_id = specification(profile)["id"]
        super().__init__(budget, directory, kind="judge", max_calls=max_calls,
                         progress=progress, sleep=sleep, clock=clock, gap=gap, actor_model=actor_model)
        self._local = threading.local()
        self._scopes = {}
        self._scope_counter = 0
        self._last_error_type = None

    # Validate the pinned request first, then apply only the uniformly declared primary token allowance.
    def _body(self, args, kwargs) -> dict:
        body = super()._body(args, kwargs)
        body["max_tokens"] = effective_max_tokens(body["model"], body["max_tokens"], self.profile)
        return body

    # Start one parser or direct-text scope without storing its private prompt a second time.
    def begin_scope(self, kind: str, model: str, prompt: str, max_tokens: int) -> int:
        with self.lock:
            if self.halted:
                raise native.InferenceHalt("The judgment transport is already halted")
            self._scope_counter += 1
            number = self._scope_counter
            record = {"schema": "aos-judge-response-scope-v1", "scope": number,
                      "kind": kind, "status": "started", "profile": self.profile_id,
                      "model": model, "max_tokens": effective_max_tokens(model, max_tokens, self.profile),
                      "upstream_requested_max_tokens": max_tokens,
                      "effective_max_tokens": effective_max_tokens(model, max_tokens, self.profile),
                      "output_allowance_amended": effective_max_tokens(model, max_tokens, self.profile) != max_tokens,
                      "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                      "maximum_responses": JSON_ATTEMPTS * EMPTY_TEXT_ATTEMPTS
                      if kind == "primary_json" else EMPTY_TEXT_ATTEMPTS,
                      "calls": []}
            self._scopes[number] = record
            write_record(self.directory / "response-scopes" / f"{number:04d}.json", record)
            self.progress(f"[judge scope {number}] Starting {kind}; model={model}; "
                          f"requested_max_tokens={max_tokens}; effective_max_tokens={effective_max_tokens(model, max_tokens, self.profile)}")
            return number

    # Persist a parser outcome separately from the successful delivery of its earlier candidates.
    def finish_scope(self, number: int, *, accepted: bool, error_type: str | None = None):
        with self.lock:
            record = self._scopes[number]
            record.update(status="accepted" if accepted else "failed", responses=len(record["calls"]))
            if error_type:
                record["error_type"] = error_type
            if not accepted:
                self.halted = True
            write_record(self.directory / "response-scopes" / f"{number:04d}.json", record, replace=True)
            outcome = "Accepted" if accepted else "Failed"
            self.progress(f"[judge scope {number}] {outcome} after {len(record['calls'])} received/attempted candidates"
                          + (f" ({error_type})" if error_type else ""))

    # List immutable scope receipts independently of actual provider-completion receipts.
    def response_scopes(self) -> list[dict]:
        return [{"path": str(path.relative_to(self.directory)),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted((self.directory / "response-scopes").glob("*.json"))]


    # Summarize a retained failure using fixed metadata rather than private prompt or response text.
    def failure_summary(self) -> dict:
        paths = sorted((self.directory / "calls").glob("*.json"), key=lambda path: int(path.stem))
        last = json.loads(paths[-1].read_text(encoding="utf-8")) if paths else {}
        failed_scopes = [value for value in self._scopes.values() if value["status"] == "failed"]
        scope = failed_scopes[-1] if failed_scopes else {}
        observation = last.get("response_observation", {})
        raw = last.get("response")
        choices = raw.get("choices") if isinstance(raw, dict) else []
        choices = choices if isinstance(choices, list) else []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        return {"profile": self.profile_id, "code": self._last_error_type or scope.get("error_type") or "JudgmentIncomplete",
                "scope": scope.get("scope"), "scope_responses": len(scope.get("calls", [])),
                "model_calls": self.usage["calls"], "unpriced_calls": self.usage["unpriced_calls"],
                "response_cost_usd": self.usage["cost_usd"],
                "last_call_receipt": str(paths[-1].relative_to(self.directory)) if paths else None,
                "last_finish_reason": finish_label(observation.get("finish_reason", choice.get("finish_reason"))),
                "last_native_finish_reason": finish_label(observation.get("native_finish_reason", choice.get("native_finish_reason"))),
                "last_response_acceptance": last.get("response_acceptance"),
                "last_upstream_requested_max_tokens": last.get("upstream_requested_max_tokens"),
                "last_effective_max_tokens": last.get("request", {}).get("max_tokens")}

    # Retain historical content handling while rejecting unsupported provider envelope shapes.
    def _response(self, raw, body):
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise native.NativeError("Judge completion returned an invalid choices envelope")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise native.NativeError("Judge completion returned an invalid message envelope")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise native.NativeError("Judge completion returned a nontext message")
        # The upstream judge reads only content; reasoning is retained in the private raw receipt.
        exposed = deepcopy(message)
        exposed["content"] = content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(**exposed))])

    # Make one metered dispatch; only a completely received and priced candidate can be returned upstream.
    def __call__(self, *args, **kwargs):
        with self.lock:
            if self.halted:
                raise native.InferenceHalt("The judgment transport is already halted")
            path = record = None
            started = time.monotonic()
            try:
                if self.usage["calls"] >= self.max_calls:
                    raise native.NativeError("Stage call bound reached; no inference started")
                body = self._body(args, kwargs)
                delay = self.ready_at - self.clock()
                if delay > 0:
                    self.sleep(delay)
                before = self.budget.admit()
                number = self.usage["calls"] + 1
                scope = getattr(self._local, "scope", None)
                candidate = None
                if scope is not None:
                    scoped = self._scopes[scope]
                    if len(scoped["calls"]) >= scoped["maximum_responses"]:
                        raise native.NativeError("The fixed response-scope bound was exceeded")
                    scoped["calls"].append(number)
                    candidate = len(scoped["calls"])
                    write_record(self.directory / "response-scopes" / f"{scope:04d}.json", scoped, replace=True)
                path = self.directory / "calls" / f"{number:03d}.json"
                record = {"call": number, "status": "started", "request": body,
                          "meter_before": before, "provider_cost_usd": None,
                          "inference_retry_policy": self.profile_id,
                          "upstream_requested_max_tokens": kwargs["max_tokens"],
                          "effective_max_tokens": body["max_tokens"],
                          "output_allowance_amended": kwargs["max_tokens"] != body["max_tokens"],
                          "upstream_requested_timeout_seconds": kwargs["timeout"],
                          "http_timeout_seconds": getattr(self.budget.transport, "timeout", None),
                          "response_scope": scope, "scope_candidate": candidate}
                write_record(path, record)
                self.usage["calls"] += 1
                scope_label = f"; scope {scope} candidate {candidate}" if scope is not None else ""
                self.progress(f"[judge call {number}/{self.max_calls}{scope_label}] Sending {body['model']}")
                raw = self.budget.transport.request("POST", "/api/v1/chat/completions", body)
                self.ready_at = self.clock() + self.gap
                record["response"] = raw
                usage = raw.get("usage") or {}
                cost = amount(usage.get("cost"), "provider response cost")
                if usage.get("is_byok") is True:
                    raise native.NativeError("External BYOK costs are unsupported")
                record["provider_cost_usd"] = cost
                self.budget.record_cost(cost)
                self.usage["cost_usd"] = round(self.usage["cost_usd"] + cost, 12)
                for key, target in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
                    value = usage.get(key, 0)
                    if type(value) is not int or value < 0:
                        raise native.NativeError("Invalid provider token count")
                    self.usage[target] += value
                details = usage.get("completion_tokens_details") or {}
                reasoning = details.get("reasoning_tokens") or 0
                if type(reasoning) is not int or reasoning < 0:
                    raise native.NativeError("Invalid provider reasoning token count")
                self.usage["reasoning_tokens"] += reasoning
                response = self._response(raw, body)
                choice = raw["choices"][0]
                content = response.choices[0].message.content or ""
                primary = body["model"] == native.JUDGES[0]
                parseable = primary_json_usable(content) if primary else None
                acceptance = ("retryable_empty" if not content.strip() else
                              "retryable_primary_json" if primary and not parseable else
                              "upstream_candidate")
                record.update(response_acceptance=acceptance,
                              response_observation={"finish_reason": choice.get("finish_reason"),
                                  "native_finish_reason": choice.get("native_finish_reason"),
                                  "non_normal_finish": choice.get("finish_reason") not in {"stop", "end_turn"},
                                  "content_characters": len(content), "primary_json_usable": parseable},
                              status="ok", meter_after=self.budget.snapshot(),
                              elapsed_seconds=round(time.monotonic() - started, 3))
                write_record(path, record, replace=True)
                finish = finish_label(choice.get("finish_reason"))
                self.progress(f"[judge call {number}/{self.max_calls}] Received; finish={finish}; "
                              f"acceptance={acceptance}; response charge ${cost:.6f}")
                if acceptance.startswith("retryable_"):
                    self.progress("[judge response] Unusable candidate retained; the bounded upstream response loop decides whether to call again")
                return response
            except BaseException as exc:
                self.halted = True
                self._last_error_type = type(exc).__name__
                if record is not None:
                    record.update(status="failed", error_type=type(exc).__name__,
                                  elapsed_seconds=round(time.monotonic() - started, 3))
                    if record["provider_cost_usd"] is None:
                        self.usage["unpriced_calls"] += 1
                    write_record(path, record, replace=True)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise native.InferenceHalt(f"Judge transport stopped ({type(exc).__name__}); no transport retry") from None


# Let the unchanged retry loops run, while preventing their final empty fallbacks from becoming scores.
@contextmanager
def guarded_responses(judges, adapter: MeteredJudgeCompletion):
    original_call, original_json = judges._call, judges._call_json

    # Run one complete historical parser scope and reject exhausted JSON instead of accepting its fallback.
    def call_json(model, api_key, prompt, max_tokens=4096):
        previous = getattr(adapter._local, "scope", None)
        scope = adapter.begin_scope("primary_json", model, prompt, max_tokens)
        adapter._local.scope = scope
        accepted = False
        error_type = None
        try:
            parsed, raw = original_json(model, api_key, prompt, max_tokens)
            if not parsed:
                error_type = "PrimaryJSONExhausted"
                raise native.InferenceHalt("Judge JSON exhausted its five fixed parser attempts")
            accepted = True
            return parsed, raw
        except BaseException as exc:
            error_type = error_type or type(exc).__name__
            raise
        finally:
            adapter._local.scope = previous
            adapter.finish_scope(scope, accepted=accepted, error_type=error_type)

    # Preserve nested primary empty retries and reject an exhausted direct-text/verifier scope.
    def call(model, api_key, prompt, max_tokens=16384):
        if getattr(adapter._local, "scope", None) is not None:
            return original_call(model, api_key, prompt, max_tokens)
        scope = adapter.begin_scope("text", model, prompt, max_tokens)
        adapter._local.scope = scope
        accepted = False
        error_type = None
        try:
            raw = original_call(model, api_key, prompt, max_tokens)
            if not isinstance(raw, str) or not raw.strip():
                error_type = "TextResponseExhausted"
                raise native.InferenceHalt("Judge text response exhausted its five fixed empty-content attempts")
            accepted = True
            return raw
        except BaseException as exc:
            error_type = error_type or type(exc).__name__
            raise
        finally:
            adapter._local.scope = None
            adapter.finish_scope(scope, accepted=accepted, error_type=error_type)

    judges._call, judges._call_json = call, call_json
    try:
        yield
    finally:
        judges._call, judges._call_json = original_call, original_json


# Load exact upstream source with temporary exhaustion guards and no change to source files.
@contextmanager
def guarded_loaded(upstream, adapter: MeteredJudgeCompletion):
    with native._loaded(upstream, adapter) as modules:
        with guarded_responses(modules["judges"], adapter):
            yield modules


# Require every intended audit surface while preserving the evaluator's semantic scores.
def judgment_health(judgment: dict, pipeline: dict, scenario_root: Path, *,
                    actor_model=native.ACTOR_MODEL, profile="capacity") -> dict:
    config, health = judgment["config"], judgment["verification"]
    expected = {"scenario": scenario_root.name, "system": pipeline["config"]["system"],
                "agent_llm": actor_model, "judge_llm": native.JUDGES[0], "judge_mode": "chain",
                "verifier_llm_1": native.JUDGES[1], "verifier_llm_2": native.JUDGES[2],
                "verifier_llm_3": native.JUDGES[3], "roster_in_judges": True,
                "violation_leniency": True, "output_audit": True,
                "agent_audit": True, "memory_audit": False}
    if any(config.get(key) != value for key, value in expected.items()):
        raise native.NativeError("Official judgment settings or pipeline identity changed")
    if Path(config.get("scenario_dir", "")).resolve() != scenario_root.resolve():
        raise native.NativeError("Official judgment scenario path changed")
    if (health.get("degraded") is not False or health.get("votes_missing") != 0
            or health.get("abstained") != 0 or type(health.get("checks")) is not int
            or judgment.get("V_vis") is None
            or type(judgment.get("utility", {}).get("correctness")) is not bool
            or judgment.get("efficiency") != {"time": pipeline["pipeline_time"], "rounds": pipeline["n_a2a"]}):
        raise native.NativeError("Incomplete or degraded official judgment")
    expected_agents = {name for name, content in pipeline.get("agent_contents", {}).items()
                       if isinstance(content, str) and content.strip()}
    agents = judgment.get("V_A")
    if not isinstance(agents, dict) or set(agents) - {"total"} != expected_agents:
        raise native.NativeError("Official judgment omitted an expected agent-context audit")
    for name in ("V_G", "V_out", "utility", "completeness"):
        if not isinstance(judgment.get(name), dict):
            raise native.NativeError("Official judgment omitted a required output audit")
    if pipeline.get("log") and not isinstance(judgment.get("V_A2A"), dict):
        raise native.NativeError("Official judgment omitted the recorded communication audit")
    return {"verification": deepcopy(health), "agent_audit": True,
            "expected_agent_audits": len(expected_agents),
            "communication_audit_required": bool(pipeline.get("log")),
            "response_profile": specification(profile)["id"]}


# Execute a fresh judging attempt with isolated upstream modules and unchanged canonical evaluator output.
def _evaluate_private(scenario_root, pipeline, out, budget, *, actor_model, profile, max_calls, progress):
    upstream = native.VENDOR_ROOT
    upstream = Path(upstream).expanduser().resolve(strict=True)
    scenario_root = Path(scenario_root).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve()
    native.source_fingerprint(upstream)
    if out.exists():
        raise native.NativeError("Choose a new judging directory; saved attempts are never overwritten")
    out.mkdir(parents=True, mode=0o700)
    console, errors = sys.stdout, sys.stderr

    # Show only fixed operational progress while retaining raw upstream output in the private log.
    def notice(message):
        with redirect_stdout(console), redirect_stderr(errors):
            progress(message)

    adapter = MeteredJudgeCompletion(budget, out, max_calls=max_calls, progress=notice,
                                     profile=profile, actor_model=actor_model)
    stage_root = out / "upstream"
    stage_root.mkdir(mode=0o700)
    write_record(out / "profile.json", {"schema": "aos-judge-response-profile-v1",
                 "profile": specification(profile), "pipeline_content_sha256": digest(pipeline)})
    old_umask = os.umask(0o077)
    timeout = getattr(budget.transport, "timeout", 120)
    budget.transport.timeout = specification(profile)["judge_http_timeout_seconds"]
    try:
        with (out / "upstream.log").open("x", encoding="utf-8") as transcript:
            with redirect_stdout(transcript), redirect_stderr(transcript), guarded_loaded(upstream, adapter) as modules:
                runner = modules["run_evaluation"]
                args = runner.build_parser().parse_args(
                    ["-d", str(scenario_root), "-o", str(stage_root), *native.JUDGE_ARGS])
                write_record(stage_root / scenario_root.name / f"pipeline_{scenario_root.name}.json", pipeline)
                runner.run_single(args, native._PLACEHOLDER_KEY)
        result_path = stage_root / scenario_root.name / f"evaluation_{scenario_root.name}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        health = judgment_health(result, pipeline, scenario_root, actor_model=actor_model, profile=profile)
        if adapter.halted or not adapter.usage["calls"]:
            raise native.NativeError("Judgment produced no successful completion")
        return {"judgment": result, "usage": deepcopy(adapter.usage),
                "receipts": adapter.receipts(), "response_scopes": adapter.response_scopes(),
                "profile": specification(profile), "health": health,
                "upstream_output": str(result_path),
                "upstream_output_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()}
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        failure = native.NativeError(f"Judgment stopped ({type(exc).__name__}); preserve this attempt",
                                     usage=adapter.usage, receipts=adapter.receipts())
        failure.response_scopes = adapter.response_scopes()
        failure.failure_summary = adapter.failure_summary()
        raise failure from None
    finally:
        budget.transport.timeout = timeout
        os.umask(old_umask)


# Judge one completed pipeline using its fixed actor identity and evaluator profile.
def evaluate(scenario_root, pipeline: dict, out, budget, *, actor_model=native.ACTOR_MODEL,
             profile="capacity", max_calls=2048, progress=print) -> dict:
    specification(profile)
    if (not isinstance(pipeline, dict) or pipeline.get("config", {}).get("system")
            not in {"decentralized", "aos_artifact_requests"}
            or pipeline.get("config", {}).get("agent_llm") != actor_model):
        raise native.NativeError("The pipeline does not match the selected actor")
    if type(max_calls) is not int or not 1 <= max_calls <= 10000:
        raise native.NativeError("The judge call ceiling must be bounded")
    scenario_root = Path(scenario_root).expanduser().resolve(strict=True)
    raw = (scenario_root / "scenario.json").read_bytes()
    scenario = json.loads(raw)
    config = pipeline["config"]
    if (config.get("scenario") != scenario_root.name
            or pipeline.get("task_description") != scenario["task"]["description"]):
        raise native.NativeError("The pipeline task does not match the evaluator input")
    if config["system"] == "decentralized":
        native.pipeline_health(pipeline, scenario_root, actor_model=actor_model)
    elif config.get("source", {}).get("scenario_sha256") != hashlib.sha256(raw).hexdigest():
        raise native.NativeError("The AOS pipeline is not bound to the selected scenario bytes")
    with native._UPSTREAM_LOCK:
        return _evaluate_private(scenario_root, deepcopy(pipeline), out, budget,
                                 actor_model=actor_model, profile=profile,
                                 max_calls=max_calls, progress=progress)
