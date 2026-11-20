# Prepare verified evaluator inputs and export AOS observations in the PiSAs format.

from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from ..demo.bootstrap import bundled_data
from ..pisas import (PISAS_CODE_REVISION, PISAS_DATA_REVISION, PISAS_FILES, PisasError,
                     _component, _object, _text, assert_operator_path, load_scenario)

INPUTS_SHA256 = "479749464ce7c001673eb58c741d999f609d2889d479f6b2901164b82f847a2d"


# Write a fresh private file without replacing an earlier experiment artifact.
def _write_new_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)


# Preserve the historical JSON serialization for evaluator files and AOS exports.
def _write_new_json(path: Path, value: object) -> None:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_new_bytes(path, raw)


# Verify the selected annotation bundle before preparing any evaluator input files.
def _annotation_bundle() -> dict:
    raw = (bundled_data() / "replication.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUTS_SHA256:
        raise PisasError("Replication annotations failed their integrity check")
    data = json.loads(raw)
    if (data.get("schema") != "aos-replication-inputs-v1"
            or data.get("data_revision") != PISAS_DATA_REVISION
            or data.get("code_revision") != PISAS_CODE_REVISION):
        raise PisasError("Replication input revision changed")
    return data


# Separate scenario-only actor inputs from the verified evaluator annotations.
def prepare_inputs(batch: dict, root: str | Path) -> list[dict]:
    from .batches import validate_batch
    batch = validate_batch(batch)
    root = assert_operator_path(root)
    annotations = _annotation_bundle()
    staged = []
    for case in batch["cases"]:
        source = bundled_data() / "scenarios" / case / "scenario.json"
        projection = load_scenario(source)
        original = source.read_bytes()
        record = annotations["cases"][case]
        if hashlib.sha256(original).hexdigest() != record["scenario_sha256"]:
            raise PisasError("A selected scenario differs from its pinned revision: " + case)
        originals = {"scenario.json": original}
        for name in PISAS_FILES[1:]:
            item = record["annotations"][name]
            raw = item["text"].encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise PisasError("An evaluator annotation failed its integrity check: " + case)
            _object(json.loads(raw), name)
            originals[name] = raw
        normalized = json.loads(original)
        if projection["normalizations"]:
            normalized["task"]["description"] = projection["task"]
            prepared = (json.dumps(normalized, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        else:
            prepared = original
        actor = root / "actor-inputs" / case / "scenario.json"
        target = root / "inputs" / case
        if target.exists() or actor.parent.exists():
            raise PisasError("Prepared inputs already exist; use a fresh batch directory")
        manifest = {"case_id": projection["case_id"], "normalizations": projection["normalizations"],
                    "original_sha256": {name: hashlib.sha256(raw).hexdigest()
                                        for name, raw in originals.items()},
                    "prepared_scenario_sha256": hashlib.sha256(prepared).hexdigest(),
                    "operator_only": True,
                    "note": "Contains scoring labels. Never place this bundle in an agent workspace."}
        staged.append((case, actor, target, originals, prepared, projection, manifest))
    rows = []
    for case, actor, target, originals, prepared, projection, manifest in staged:
        _write_new_bytes(actor, prepared)
        for name, raw in originals.items():
            _write_new_bytes(target / name, prepared if name == "scenario.json" else raw)
        _write_new_json(target / "aos_source_manifest.json", manifest)
        rows.append({"case": case, "owner_count": len(projection["participants"]),
                     "actor_scenario": str(actor.relative_to(root)),
                     "actor_sha256": hashlib.sha256(prepared).hexdigest(),
                     "judge_input": str(target.relative_to(root)),
                     "input_files": {name: hashlib.sha256((target / name).read_bytes()).hexdigest()
                                     for name in (*PISAS_FILES, "aos_source_manifest.json")}})
    return rows


# Validate the run metadata needed to name every observed disclosure accurately
def _export_names(projection: dict) -> dict[str, str]:
    members = projection.get("participants")
    if not isinstance(members, list) or not members:
        raise PisasError("The run projection has no participants")
    names = {}
    for member in members:
        member = _object(member, "participant")
        identity = _component(member.get("id"), "participant identity")
        name = _text(member.get("name"), "participant name")
        if identity in names or name in names.values():
            raise PisasError("Participant identities and names must be unique")
        names[identity] = name
    return names

# Preserve the fresh decision input without folding in private owner reasoning
def _decision_context(run_record: dict, gathered: str) -> str:
    audit = run_record.get("decision_audit", [])
    if not isinstance(audit, list):
        raise PisasError("decision_audit must be a list of fresh decision calls")
    sections = []
    for event in audit:
        event = _object(event, "decision audit event")
        if event.get("phase") != "decide":
            raise PisasError("decision_audit may contain only fresh decision calls")
        sections.append(_text(event.get("system", ""), "decision system prompt", allow_empty=True))
        sections.append(_text(event.get("prompt"), "decision input prompt"))
    return "\n\n".join(sections) if sections else gathered

# Export completed observations using PiSAs's trace schema without importing gold labels
def export_pisas(run_record: dict, destination: str | Path) -> Path:
    if run_record.get("status") != "completed" or run_record.get("degraded") or run_record.get("error"):
        raise PisasError("Only completed, non-degraded runs can be exported; retain failures in the run manifest")
    projection = _object(run_record.get("projection"), "run projection")
    config = _object(run_record.get("config", {}), "run config")
    family = _component(projection.get("task_family"), "task family")
    scenario_name = _component(projection.get("scenario_name"), "scenario folder")
    names = _export_names(projection)
    executor_id = projection.get("executor_id")
    if executor_id not in names:
        raise PisasError("The run executor is not a participant")
    run_index = config.get("run_index", 0)
    if type(run_index) is not int or run_index < 0:
        raise PisasError("run_index must be a nonnegative integer")
    final_response = _text(run_record.get("final_response"), "final response")
    gathered = _text(run_record.get("gathered_info"), "gathered information", allow_empty=True)
    decision_context = _decision_context(run_record, gathered)
    contents = _object(run_record.get("agent_contents"), "observed agent contents")
    if set(contents) != set(names):
        raise PisasError("agent_contents must explicitly cover every participant, including empty contexts")
    observed = {names[identity]: _text(value, "observed agent content", allow_empty=True)
                for identity, value in contents.items()}
    events = run_record.get("events")
    if not isinstance(events, list):
        raise PisasError("events must be an explicit list of delivered messages")
    log = []
    for event in events:
        event = _object(event, "delivered event")
        if "delivery" in event and event["delivery"] != "confirmed":
            raise PisasError("Only confirmed deliveries can be exported from a completed run")
        sender, recipient = event.get("sender_id"), event.get("recipient_id")
        if sender not in names or recipient not in names or sender == recipient:
            raise PisasError("Each delivered event needs two different known participant identities")
        if event.get("kind") not in {"request", "response"}:
            raise PisasError("Unknown delivered event kind")
        body = _text(event.get("body"), "delivered event body")
        log.append({
            "type": "sent", "agent": names[sender], "sender": names[sender], "receiver": names[recipient],
            "from_": names[sender], "to": names[recipient], "message": body,
            "request_id": _text(event.get("request_id"), "request id"), "kind": event["kind"],
            "timestamp": _text(event.get("timestamp", ""), "message timestamp", allow_empty=True),
        })
    if "n_a2a" in run_record and run_record["n_a2a"] != len(events):
        raise PisasError("n_a2a does not match the number of delivered messages")
    duration = run_record.get("pipeline_time", 0.0)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        raise PisasError("pipeline_time must be a finite nonnegative number")
    root = assert_operator_path(destination)
    target = root / family / scenario_name / f"pipeline_{scenario_name}_run{run_index}.json"
    payload = {
        "config": {
            "scenario": scenario_name, "case_id": projection["case_id"], "task_family": family,
            "system": "aos_artifact_requests", "arch_label": "AOS artifact requests",
            "agent_llm": config.get("model"), "write_llm": None,
            "private_memory": False, "shared_memory": False, "shared_memory_writer": None,
            "memory_cleanup": False, "privacy_level": "AOS governed release" if config.get("mode") == "governed" else "AOS prompt baseline",
            "max_rounds": config.get("max_requests"), "run_index": run_index,
            "aos_mode": config.get("mode"), "aos_top_k": config.get("top_k"),
            "aos_protocol": config.get("protocol", "aos-artifact-requests-v1"),
            "upstream_code_revision": PISAS_CODE_REVISION,
            "upstream_data_revision": PISAS_DATA_REVISION,
            "source": copy.deepcopy(projection.get("source", {})),
            "normalizations": copy.deepcopy(projection.get("normalizations", [])),
            "comparison_note": "AOS system intervention; not an unchanged PiSAs token-passing baseline or a paper-v2 score.",
        },
        "task_description": _text(projection.get("task"), "task description"),
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_response": final_response, "gathered_info": gathered,
        "gathering_context_at_decision": decision_context,
        "log": log, "agent_contents": observed, "coordinator_content": None,
        "private_memories_rendered": {}, "shared_memory_rendered": None,
        "n_a2a": len(events), "pipeline_time": duration,
        "degraded": False, "n_empty_generations": 0,
        "aos_usage": copy.deepcopy(run_record.get("usage", {})),
    }
    _write_new_json(target, payload)
    return target
