# Run unchanged owner components and an explicit prospective answer adapter with faithful raw model audits.

from __future__ import annotations
import fcntl
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode
from ...llm import add_usage
from ..common import canonical, read_json, timestamp, write_json
from .runtime import (IntegratedService, OWNER_REQUEST_TIMEOUT_SECONDS, _CallGate,
                                  _settled_audits, _stop_services, _validate_prepared)
from ..model import call_json, usage_delta, usage_snapshot, visible_context
from ..runtime import RunFailed, ServiceGroup, _record_decision, request_json, validate_response
from ..service import ArtifactService
from . import contract


# Execute actual authenticated exchanges and persist the native decision task after owner locks close.
def run(root: str | Path, model, *, treatment: str = "integrated", progress=None) -> dict:
    root = Path(root).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    _validate_prepared(root, manifest, treatment, model)
    if manifest["config"].get("response_contract") != contract.specification():
        raise RunFailed("Prepared experiment lacks the exact prospective response contract")
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
              "decision_contract": {"id": contract.CONTRACT_ID, "status": "not_started"},
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
            result["decision_contract"]["status"] = "calling_model"
            raw_decision = call_json(gated_model, "decide", {
                "task": projection["task"], "recipient": projection["recipient"],
                "participants": projection["participants"], "gathered_info": result["gathered_info"],
                "evidence_refs": refs,
            }, result["decision_audit"])
            result["decision_contract"]["status"] = "validating"
            decision, result["decision_contract"] = contract.adapt(raw_decision, set(refs))
        except BaseException as exc:
            result["decision_contract"].update(status="rejected", error_type=type(exc).__name__)
            raise
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
