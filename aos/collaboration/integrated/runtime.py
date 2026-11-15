# Run the selected owner components through real authenticated AOS artifact services.

from __future__ import annotations
import fcntl
import json
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode
from ...llm import add_usage
from ...tasks import Step, Task
from ..artifacts import EvidenceBundle
from ..common import canonical, read_json, timestamp, write_json
from ..model import REVIEW, call_json, usage_delta, usage_snapshot, visible_context
from ..policy import RequestContext
from ..release import ReleaseProposal, ReviewRecord, release
from .components import (CandidateError, MAX_INPUT_BYTES, _call, _shape,
                         extraction_system, NORMALIZE, normalize_proposal)
from ..runtime import RunFailed, ServiceGroup, _decision, _record_decision, request_json, validate_response
from ..service import ArtifactService, RequestFailure

TREATMENTS = ("governed-v1", "integrated")
MAX_OWNER_EVIDENCE_UNITS = 2048
OWNER_REQUEST_TIMEOUT_SECONDS = 1800


# Prevent a timed-out or interrupted coordinator from starting a later inference phase.
class _CallGate:
    # Wrap the injected serial model without changing its settings, prompts, or accounting.
    def __init__(self, model, cancelled: threading.Event):
        self.model, self.cancelled = model, cancelled

    # Expose the original usage counters to the unchanged AOS accounting helpers.
    @property
    def usage(self):
        return getattr(self.model, "usage", {})

    # Permit one model invocation only while the coordinator still owns this run.
    def complete_json(self, prompt, system=None):
        if self.cancelled.is_set():
            raise RunFailed("Coordinator stopped; no additional inference phase started")
        return self.model.complete_json(prompt, system=system)


# Reject an oversized complete owner input instead of silently selecting a prefix.
def _bounded_payload(payload: dict) -> int:
    if len(payload["evidence"]) > MAX_OWNER_EVIDENCE_UNITS:
        raise CandidateError("Complete owner corpus exceeds the evidence-unit bound")
    size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    if size > MAX_INPUT_BYTES:
        raise CandidateError("Complete owner payload exceeds the one-megabyte bound")
    return size


# Extend only owner processing while retaining authentication, idempotency, and workspace locks.
class IntegratedService(ArtifactService):
    # Require the already prepared integrated treatment before creating any service state.
    def __init__(self, config_path: str | Path, model):
        config = read_json(Path(config_path))
        if config.get("top_k") is not None or config.get("mode") != "governed":
            raise ValueError("Integrated owners require all-owner retrieval and governed review")
        if model is None:
            raise ValueError("The integrated runtime requires an explicitly metered model")
        super().__init__(config_path, model=model)

    # Extract, normalize, review, and release only current owner-local supported claims.
    def _process(self, principal: str, data: dict) -> dict:
        context = RequestContext(data["request_id"], principal, self.owner_id,
                                 self.config["task"], data["question"])
        audit = {"request_id": context.request_id, "context": context.as_dict(),
                 "status": "processing", "events": [], "started": timestamp(),
                 "treatment": "integrated"}
        task = Task.new(self.ws, "Answer an authenticated artifact request")
        record = task.start_step(Step(worker="artifact_request", instructions=data["question"],
                                     reason="Retrieve all bounded owner evidence and normalize before review"))
        before = usage_snapshot(self.model)
        task.log_path(1).write_text(
            f"Artifact request {context.request_id}\nRequester: {principal}\n"
            "Treatment: all-owner retrieval, cap24 extraction, operational rewrite, unchanged review\n",
            encoding="utf-8")
        try:
            units = self.index.search(data["question"], limit=None)
            bundle = EvidenceBundle.create(context, units)
            policy = self.policy.load()
            if policy.mode != "inferred":
                raise CandidateError("The tested operational rewrite requires an inferred owner policy")
            record["inputs"] = sorted({unit.source_path for unit in units})
            task.save()
            audit["retrieved"] = bundle.as_dict()
            audit["policy"] = policy.as_dict()
            payload = {"context": context.as_dict(), "participants": self.config["participants"],
                       "relations": self.config.get("relations", []),
                       "policy_instructions": policy.instructions,
                       "evidence": [unit.as_dict() for unit in units]}
            audit["retrieval"] = {"strategy": "all", "unit_count": len(units),
                                  "maximum_units": MAX_OWNER_EVIDENCE_UNITS,
                                  "maximum_payload_bytes": MAX_INPUT_BYTES,
                                  "truncated": False}
            audit["retrieval"]["payload_bytes"] = _bounded_payload(payload)
            extracted = (_call(self.model, "extract", extraction_system("cap24"), payload,
                               audit["events"]) if units else {"claims": []})
            _shape(extracted, {"claims"}, "extraction")
            parent = ReleaseProposal.create(context, bundle, policy, extracted["claims"])
            audit["parent_proposal"] = parent.as_dict()
            audit["extraction"] = {"prompt_claim_limit": 24, "proposal_count": len(parent.claims),
                                   "prompt_limit_complied": len(parent.claims) <= 24}
            proposal = parent
            review_payload = {key: payload[key] for key in
                              ("context", "participants", "relations", "policy_instructions")}
            if parent.claims:
                review_payload["proposal"] = parent.as_dict()
                audit["normalization"] = {"status": "started", "parent_digest": parent.digest()}
                normalized = _call(self.model, "normalize_release", NORMALIZE, review_payload,
                                   audit["events"])
                # Preserve the raw result even if lineage validation rejects it before review.
                audit["normalization"]["response"] = normalized
                proposal, lineage = normalize_proposal(
                    {"payload": payload, "policy": policy.as_dict()}, parent, normalized)
                audit["normalization"].update(status="completed", lineage=lineage,
                                              child_digest=proposal.digest())
            else:
                audit["normalization"] = {"status": "not_needed", "reason": "empty_proposal",
                                          "parent_digest": parent.digest(), "child_digest": parent.digest(),
                                          "lineage": []}
            audit["proposal"] = proposal.as_dict()
            review_payload["proposal"] = proposal.as_dict()
            reviewed = (_call(self.model, "review", REVIEW, review_payload, audit["events"])
                        if proposal.claims else {"decisions": []})
            review = ReviewRecord.from_dict(proposal, reviewed)
            audit["review"] = review.as_dict()
            # Reuse the current-source and current-policy release gate rather than an offline response builder.
            response = release(self.index, self.policy, context, proposal, review).as_dict()
            relative = f"generated/responses/{context.request_id}.json"
            write_json(self.ws.abspath(relative), response)
            self.ws.register(relative, source="artifact_request", agent=self.owner_id, task=task.id,
                             step=1, reason="Canonical checked artifact response",
                             summary="Released claims and opaque evidence references", inputs=record["inputs"])
            usage = usage_delta(self.model, before)
            task.finish_step(record, status="ok", outputs=[relative],
                             summary=f"Artifact response: {response['status']}", usage=usage)
            task.finish("done", response["answer"] or "No releasable evidence", usage=usage)
            record["commit"] = self.ws.checkpoint(f"{task.id}: artifact response committed") or self.ws.head()
            task.save()
            audit.update(status="completed", response=response, task_id=task.id, usage=usage,
                         agent_content=visible_context(audit["events"]), finished=timestamp())
            write_json(self.directory / "audit" / f"{context.request_id}.json", audit)
            self._finish(context.request_id, "completed", response, task.id)
            return response
        except BaseException as exc:
            usage = usage_delta(self.model, before)
            code = type(exc).__name__
            if audit.get("normalization", {}).get("status") == "started":
                audit["normalization"].update(status="failed", error_type=code)
            task.finish_step(record, status="failed", outputs=[], summary="", error=code, usage=usage)
            task.finish("failed", f"Artifact request failed ({code})", usage=usage)
            self.ws.checkpoint(f"{task.id}: artifact request failed")
            audit.update(status="failed", error_type=code, task_id=task.id, usage=usage,
                         agent_content=visible_context(audit["events"]), finished=timestamp())
            write_json(self.directory / "audit" / f"{context.request_id}.json", audit)
            self._finish(context.request_id, "failed", None, task.id)
            raise RequestFailure(500, "request_failed") from None


# Check treatment identity without altering the prepared manifest or frozen incumbent modules.
def _validate_prepared(root: Path, manifest: dict, treatment: str, model) -> None:
    if treatment not in TREATMENTS or model is None:
        raise ValueError("A known treatment and an explicit metered model are required")
    if (root / "run.json").exists():
        raise RunFailed("This experiment already has a run; no inference will be repeated")
    expected_top_k = None if treatment == "integrated" else 8
    config = manifest["config"]
    if config.get("mode") != "governed" or config.get("top_k") != expected_top_k:
        raise RunFailed("Prepared retrieval/review settings do not match the selected treatment")
    expected_version = "integrated-v3" if treatment == "integrated" else "governed-v1"
    if config.get("system_version", expected_version) != expected_version:
        raise RunFailed("Prepared system version differs from the selected treatment")
    for owner in manifest["owners"]:
        settings = read_json(root / owner["config"])
        if settings.get("mode") != "governed" or settings.get("top_k") != expected_top_k:
            raise RunFailed("An owner's prepared settings differ from the selected treatment")


# Stop listeners and wait for active model receipts before allowing any later run.
def _stop_services(services: list[ArtifactService], cancelled: threading.Event) -> None:
    cancelled.set()
    for service in reversed(services):
        service.stop()
    for service in services:
        if not service._busy.acquire(timeout=OWNER_REQUEST_TIMEOUT_SECONDS):
            raise RunFailed("An owner request remains unsettled; do not start another run")
        service._lock.close()
        service._busy.release()


# Recover a terminal private audit after an HTTP timeout without inventing a delivered response.
def _settled_audits(root: Path, attempts: list[tuple], collected: set[str], result: dict) -> None:
    for owner, request_id, request_event in attempts:
        if owner["id"] in collected:
            continue
        path = (root / owner["config"]).parent / "audit" / f"{request_id}.json"
        try:
            audit = read_json(path)
            if audit.get("request_id") != request_id or audit.get("status") not in {"completed", "failed"}:
                raise RunFailed("Owner audit did not settle to a terminal request")
            result["agent_contents"][owner["id"]] += audit.get("agent_content", "")
            result["owner_usage"][owner["id"]] = audit.get("usage", {})
            if request_event is not None:
                request_event["delivery"] = "confirmed"
            collected.add(owner["id"])
        except Exception:
            result["degraded"] = True


# Execute actual authenticated exchanges and persist the native decision task after owner locks close.
def run(root: str | Path, model, *, treatment: str = "integrated", progress=None) -> dict:
    root = Path(root).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    _validate_prepared(root, manifest, treatment, model)
    guard = (root / ".run.lock").open("a+")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        guard.close()
        raise RunFailed("Another coordinator owns this experiment") from None
    if (root / "run.json").exists():
        guard.close()
        raise RunFailed("This experiment already has a run; no inference will be repeated")
    projection = read_json(root / manifest["projection"])
    started = time.monotonic()
    config = dict(manifest["config"])
    config["system_version"] = "integrated-v3" if treatment == "integrated" else "governed-v1"
    result = {"status": "running", "projection": projection, "config": config,
              "events": [], "agent_contents": {person["id"]: "" for person in projection["participants"]},
              "responses": [], "final_response": "", "gathered_info": "", "n_a2a": 0,
              "started": timestamp(), "degraded": False, "owner_usage": {}, "decision_audit": [],
              "integration_treatment": treatment,
              "owner_request_timeout_seconds": OWNER_REQUEST_TIMEOUT_SECONDS}
    write_json(root / "run.json", result)
    services = []
    cancelled = threading.Event()
    gated_model = _CallGate(model, cancelled)
    service_type = IntegratedService if treatment == "integrated" else ArtifactService
    executor = manifest["executor_id"]
    attempts, collected = [], set()
    try:
        for owner in manifest["owners"]:
            service = service_type(root / owner["config"], model=gated_model)
            services.append(service)
            service.start()
        urls = ServiceGroup(root, manifest, external=True).start()
        # Preserve the incumbent protocol: each owner sees only the original task and its own artifacts.
        for owner in manifest["owners"]:
            request_id = "req-" + uuid.uuid4().hex
            request = {"request_id": request_id, "case_id": manifest["case_id"],
                       "question": projection["task"]}
            request_event = None
            if owner["id"] != executor:
                request_event = {"sender_id": executor, "recipient_id": owner["id"],
                                 "request_id": request_id, "kind": "request", "body": request["question"],
                                 "delivery": "attempted", "timestamp": timestamp()}
                result["events"].append(request_event)
                result["n_a2a"] += 1
            attempts.append((owner, request_id, request_event))
            write_json(root / "run.json", result)
            if progress:
                progress(f"Requesting artifacts from {owner['name']} ({owner['id']})")
            try:
                response = validate_response(request_json(
                    urls[owner["id"]] + "/request", owner["request_token"], request,
                    timeout=OWNER_REQUEST_TIMEOUT_SECONDS), request_id)
                if request_event is not None:
                    request_event["delivery"] = "confirmed"
            finally:
                # Privileged audit collection is evaluation-only; it never becomes a decision or peer payload.
                try:
                    audit = request_json(urls[owner["id"]] + "/audit?" + urlencode({"request_id": request_id}),
                                         owner["admin_token"], timeout=5)
                    result["agent_contents"][owner["id"]] += audit.get("agent_content", "")
                    result["owner_usage"][owner["id"]] = audit.get("usage", {})
                    if request_event is not None and audit.get("request_id") == request_id:
                        request_event["delivery"] = "confirmed"
                    collected.add(owner["id"])
                except Exception:
                    result["degraded"] = True
            result["responses"].append({"owner_id": owner["id"], "response": response})
            if owner["id"] != executor:
                body = canonical(response)
                result["events"].append({"sender_id": owner["id"], "recipient_id": executor,
                                         "request_id": request_id, "kind": "response", "body": body,
                                         "delivery": "confirmed", "timestamp": timestamp()})
                result["n_a2a"] += 1
                result["agent_contents"][executor] += "\nReceived artifact response:\n" + body
            write_json(root / "run.json", result)
            if progress:
                progress(f"{owner['name']}: {response['status']} ({len(response['claims'])} released claims)")
        if result["degraded"]:
            raise RunFailed("One or more private audit channels could not be collected")
        claims = [claim for item in result["responses"] for claim in item["response"]["claims"]]
        result["gathered_info"] = "\n".join(
            f"{claim['text']} [evidence: {', '.join(claim['evidence_refs'])}]" for claim in claims)
        refs = sorted({ref for claim in claims for ref in claim["evidence_refs"]})
        before = usage_snapshot(gated_model)
        try:
            decision = _decision(call_json(gated_model, "decide", {
                "task": projection["task"], "recipient": projection["recipient"],
                "participants": projection["participants"], "gathered_info": result["gathered_info"],
                "evidence_refs": refs,
            }, result["decision_audit"]), set(refs))
        finally:
            result["decision_usage"] = usage_delta(gated_model, before)
            result["agent_contents"][executor] += "\n" + visible_context(result["decision_audit"])
        result.update(decision)
        result["usage"] = add_usage(*result["owner_usage"].values(), result["decision_usage"])
        _stop_services(services, cancelled)
        services.clear()
        _record_decision(root, manifest, result)
        result.update(status="completed", finished=timestamp(), pipeline_time=time.monotonic() - started)
        write_json(root / "run.json", result)
        return result
    except BaseException as exc:
        try:
            _stop_services(services, cancelled)
            services.clear()
        except Exception as stop_error:
            result["settlement_error"] = type(stop_error).__name__
            result["degraded"] = True
        _settled_audits(root, attempts, collected, result)
        result.update(status="failed", error=type(exc).__name__, finished=timestamp(),
                      pipeline_time=time.monotonic() - started)
        result["usage"] = add_usage(*result["owner_usage"].values(), result.get("decision_usage"))
        write_json(root / "run.json", result)
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RunFailed(f"Run failed ({type(exc).__name__}); inspect its private run record") from exc
    finally:
        guard.close()
