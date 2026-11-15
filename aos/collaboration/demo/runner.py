# Reuse owner services unchanged while giving the optional demo decision strategy its own protocol.

from __future__ import annotations
import fcntl
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode
from ...llm import add_usage
from ..common import canonical, read_json, timestamp, write_json
from ..experiment import demo_policies, demo_projection, prepare_experiment
from ..model import create_model, usage_delta, usage_snapshot, visible_context
from ..pisas import load_scenario
from ..runtime import RunFailed, ServiceGroup, _record_decision, request_json, run_experiment, validate_response
from .decisions import (OBSERVED_PROTOCOL, OBSERVATION_SCHEMA, ObservedFixtureModel,
                        call_observed, released_catalogue)


# Execute a fresh optional observed coordinator without modifying the frozen governed-v1 runtime.
def run_observed(root: str | Path, *, model=None, progress=None, external_services: bool = False) -> dict:
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    projection = read_json(root / manifest["projection"])
    run_path = root / "run.json"
    if manifest["config"].get("protocol") != OBSERVED_PROTOCOL:
        raise RunFailed("Observed coordinator requires an observed-v2 manifest")
    if run_path.exists():
        raise RunFailed("This experiment already has a run; choose a fresh demo run")
    guard = (root / ".run.lock").open("a+")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        guard.close()
        raise RunFailed("Another coordinator owns this experiment") from None
    if run_path.exists():
        guard.close()
        raise RunFailed("This experiment already has a run")
    started = time.monotonic()
    result = {"status": "running", "projection": projection, "config": manifest["config"],
              "events": [], "agent_contents": {p["id"]: "" for p in projection["participants"]},
              "responses": [], "final_response": "", "gathered_info": "", "n_a2a": 0,
              "started": timestamp(), "degraded": False, "owner_usage": {}, "decision_audit": [],
              "released_claims": [],
              "decision_observation": {"schema": OBSERVATION_SCHEMA, "status": "not_started"}}
    write_json(run_path, result)
    group = ServiceGroup(root, manifest, external=external_services)
    executor = manifest["executor_id"]
    try:
        urls = group.start()
        decision_model = model or (ObservedFixtureModel() if manifest["config"]["backend"] == "scripted"
                                   else create_model(manifest["config"]["model"]))
        # Each owner receives the original task, never another owner's released claims or private audit.
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
            write_json(run_path, result)
            if progress:
                progress(f"Requesting artifacts from {owner['name']} ({owner['id']})")
            try:
                response = validate_response(request_json(urls[owner["id"]] + "/request",
                                                          owner["request_token"], request), request_id)
                if request_event is not None:
                    request_event["delivery"] = "confirmed"
            finally:
                # The privileged audit remains evaluation-only and never enters the decision payload.
                try:
                    audit = request_json(urls[owner["id"]] + "/audit?" + urlencode({"request_id": request_id}),
                                         owner["admin_token"], timeout=5)
                    result["agent_contents"][owner["id"]] += audit.get("agent_content", "")
                    result["owner_usage"][owner["id"]] = audit.get("usage", {})
                    if request_event is not None and audit.get("request_id") == request_id:
                        request_event["delivery"] = "confirmed"
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
            write_json(run_path, result)
            if progress:
                progress(f"{owner['name']}: {response['status']} ({len(response['claims'])} released claims)")
        if result["degraded"]:
            raise RunFailed("One or more private audit channels could not be collected")
        # The observed decision receives only the exact released claims with stable local identities.
        catalogue = released_catalogue(result["responses"])
        result["released_claims"] = catalogue
        result["gathered_info"] = "\n".join(
            f"{claim['text']} [evidence: {', '.join(claim['evidence_refs'])}]" for claim in catalogue)
        refs = sorted({ref for claim in catalogue for ref in claim["evidence_refs"]})
        result["decision_observation"]["status"] = "started"
        write_json(run_path, result)
        if progress:
            progress("Making one observed decision from released evidence; disagreements do not automatically block an answer")
        before = usage_snapshot(decision_model)
        try:
            decision, observation = call_observed(decision_model, {
                "task": projection["task"], "recipient": projection["recipient"],
                "participants": projection["participants"], "gathered_info": result["gathered_info"],
                "evidence_refs": refs,
            }, catalogue, result["decision_audit"])
        finally:
            result["decision_usage"] = usage_delta(decision_model, before)
            result["agent_contents"][executor] += "\n" + visible_context(result["decision_audit"])
        result.update(decision)
        result["decision_observation"] = observation
        result["usage"] = add_usage(*result["owner_usage"].values(), result["decision_usage"])
        group.stop()
        if not external_services:
            _record_decision(root, manifest, result)
        result.update(status="completed", finished=timestamp(), pipeline_time=time.monotonic() - started)
        write_json(run_path, result)
        if progress:
            progress(f"Decision completed: {observation['decision_action']}; {len(observation['disagreements'])} recorded tensions")
        return result
    except BaseException as exc:
        group.stop()
        if result["decision_observation"]["status"] == "started":
            result["decision_observation"]["status"] = "failed"
        result.update(status="failed", error=type(exc).__name__, finished=timestamp(),
                      pipeline_time=time.monotonic() - started)
        result["usage"] = add_usage(*result["owner_usage"].values(), result.get("decision_usage"))
        write_json(run_path, result)
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RunFailed(f"Observed run failed ({type(exc).__name__}); inspect its private run record") from exc
    finally:
        guard.close()

# Prepare and run the selected collaboration system through its recorded protocol.
def execute_case(scenario_path: Path | None, run_root: Path, *, version: str = "governed-v1",
                 model: str = "gemini-3.5-flash-lite", top_k: int | None = 8, progress=None) -> dict:
    if version not in {"governed-v1", "observed-v2", "integrated-v3"}:
        raise ValueError("Unknown collaboration system version")
    if scenario_path is None and version == "integrated-v3":
        raise ValueError("Integrated v3 requires a PiSAs scenario")
    if scenario_path is None:
        projection = demo_projection()
        backend, policies = "scripted", demo_policies(projection)
    else:
        projection = load_scenario(scenario_path, allow_task_fallback=True)
        backend, policies = "model", None
    manifest = prepare_experiment(projection, run_root, model=model, mode="governed",
                                  top_k=None if version == "integrated-v3" else top_k,
                                  backend=backend, owner_policies=policies)
    if version == "integrated-v3":
        from ..integrated import contract, contract_runtime
        manifest["config"].update(system_version="integrated-v3",
                                  response_contract=contract.specification())
        write_json(Path(run_root) / "manifest.json", manifest)
        return contract_runtime.run(run_root, create_model(model), treatment="integrated", progress=progress)
    if version == "governed-v1":
        return run_experiment(run_root, progress=progress)
    # Owner configs and release policy remain unchanged (only the coordinator contract is versioned).
    manifest["protocol"] = OBSERVED_PROTOCOL
    manifest["config"].update(protocol=OBSERVED_PROTOCOL, system_version="observed-v2",
                               decision_strategy="single_observed_decision",
                               benchmark_score="not_evaluated")
    write_json(Path(run_root) / "manifest.json", manifest)
    return run_observed(run_root, progress=progress)
