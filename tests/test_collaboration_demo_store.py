# Verify replay privacy, explicit evaluation attachment, and persistent launch identity.

from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import pytest
from aos.collaboration.demo.store import DemoStore, StoreError, write_private
from aos.collaboration.demo.view import build_view, ReplayError

SECRET = "PRIVATE_OWNER_SECRET_DO_NOT_COPY"
REF = "ref-" + "b" * 24


# Supply a minimal canonical artifact trace with one actual peer response and private fields.
def sample_run():
    response = {"request_id": "req-one", "status": "released", "answer": "Capacity is 20.",
                "claims": [{"text": "Capacity is 20.", "evidence_refs": [REF]}], "coverage": "not_established"}
    return {"status": "completed", "config": {"protocol": "aos-artifact-requests-v1", "mode": "governed", "model": "test-model", "run_index": 0, "credential": SECRET},
            "projection": {"case_id": "example__scenario_01", "task": "Can the room seat 20?", "executor_id": "agent-001",
                           "participants": [{"id": "agent-001", "name": "A", "role": SECRET}, {"id": "agent-002", "name": "B"}], "artifacts": SECRET},
            "events": [{"kind": "request", "sender_id": "agent-001", "recipient_id": "agent-002", "request_id": "req-one", "body": "Can the room seat 20?", "delivery": "confirmed", "timestamp": "2026-11-01T01:00:00Z"},
                       {"kind": "response", "sender_id": "agent-002", "recipient_id": "agent-001", "request_id": "req-one", "body": json.dumps(response), "delivery": "confirmed", "timestamp": "2026-11-01T01:00:01Z"}],
            "responses": [{"owner_id": "agent-002", "response": response}], "final_response": "Yes.", "used_evidence_refs": [REF], "insufficient_evidence": False,
            "n_a2a": 2, "decision_audit": [{"response": SECRET}], "agent_contents": SECRET, "error": SECRET}


# Write an operator fixture without altering the actor contract under test.
def save_json(path, value):
    path.write_text(json.dumps(value))
    return path


# Import only the public trace and retain it across independent registry objects.
def test_replay_import_is_private_and_persistent(tmp_path):
    source = save_json(tmp_path / "run.json", sample_run())
    before = source.read_bytes()
    store = DemoStore(tmp_path / "demo")
    record = store.add_replay(source, title="Known replay")
    snapshot = DemoStore(tmp_path / "demo").snapshot(record["run_id"])
    assert snapshot["metadata"]["mode"] == "recorded_replay"
    assert snapshot["view"]["answer"]["text"] == "Yes."
    assert SECRET not in json.dumps(snapshot)
    assert source.read_bytes() == before
    assert snapshot["view"]["evidence"][0]["claim_id"] == "agent-002:req-one:c1"
    assert snapshot["view"]["conflict_detection"]["status"] == "not_recorded"


# Keep display identities stable while the file digest changes as a live run grows.
def test_event_and_claim_ids_are_stable_across_snapshots():
    run = sample_run()
    first = build_view(run, "test-run", "a" * 64)
    second = build_view(run, "test-run", "b" * 64)
    assert first["timeline"][0]["event_id"] == second["timeline"][0]["event_id"]
    assert first["evidence"][0]["claim_id"] == second["evidence"][0]["claim_id"]
    assert first["source_sha256"] != second["source_sha256"]


# Preserve a failed actor's missing answer instead of publishing its private model draft.
def test_failed_replay_keeps_failure(tmp_path):
    run = sample_run()
    run.update(status="failed", final_response="", used_evidence_refs=[])
    store = DemoStore(tmp_path / "demo")
    record = store.add_replay(save_json(tmp_path / "run.json", run))
    with store._locked() as registry:
        registry["runs"][record["run_id"]]["generation_error_type"] = "ModelFailure"
    snapshot = store.snapshot(record["run_id"])
    assert snapshot["metadata"]["status"] == "failed"
    assert snapshot["metadata"]["generation_error_type"] == "ModelFailure"
    assert snapshot["view"]["answer"]["state"] == "not_available"
    assert SECRET not in json.dumps(snapshot)


# Reject tampered transmission records rather than choosing whichever copy is convenient.
def test_response_mismatch_is_rejected(tmp_path):
    run = sample_run()
    run["events"][1]["body"] = run["events"][1]["body"].replace("Capacity is 20.", "Capacity is 200.")
    with pytest.raises(ReplayError):
        DemoStore(tmp_path / "demo").add_replay(save_json(tmp_path / "run.json", run))


# Partial observed-v2 traces remain visible without claiming a completed assessment.
def test_partial_observation_and_version_projection(tmp_path):
    run = sample_run()
    run["config"]["protocol"] = "aos-artifact-requests-observed-v2"
    run["decision_observation"] = {"schema": "aos-decision-observation-v1", "status": "started"}
    view = build_view(run, "test-run", "a" * 64)
    assert view["conflict_detection"]["status"] == "started"
    store = DemoStore(tmp_path / "demo")
    record = store.add_replay(save_json(tmp_path / "run.json", run))
    assert record["version"] == "observed-v2"


# Do not let repeated launch clicks or a new store object create a second worker.
def test_start_is_persistent_and_idempotent(tmp_path, monkeypatch):
    calls = []
    # Record launches without executing any models or subprocesses
    def launch(self, run_id, directory):
        calls.append(run_id)
        return os.getpid()
    monkeypatch.setattr(DemoStore, "_launch", launch)
    store = DemoStore(tmp_path / "demo")
    case = store.catalog()["cases"][0]["case_id"]
    first = store.start(case, "governed-v1", "launch-one")
    repeated = DemoStore(tmp_path / "demo").start(case, "governed-v1", "launch-one")
    assert first["run_id"] == repeated["run_id"] and repeated["duplicate"]
    assert len(calls) == 1
    with pytest.raises(StoreError):
        store.start(case, "observed-v2", "launch-one")
    with pytest.raises(StoreError):
        store.start(case, "governed-v1", "launch-two")


# Preserve a vanished worker as interrupted and never automatically regenerate its actor.
def test_lost_worker_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(DemoStore, "_launch", lambda *args: 999999999)
    store = DemoStore(tmp_path / "demo")
    case = store.catalog()["cases"][0]["case_id"]
    receipt = store.start(case, "governed-v1", "launch-one")
    assert store.runs()["runs"][0]["status"] == "interrupted"
    assert store.start(case, "governed-v1", "launch-one")["run_id"] == receipt["run_id"]


# Refuse arbitrary paths and non-registered IDs from the browser.
def test_browser_identifiers_cannot_escape_store(tmp_path):
    store = DemoStore(tmp_path / "demo")
    for identifier in ("../../run.json", "run-abc", "", None):
        with pytest.raises(StoreError):
            store.snapshot(identifier)
    with pytest.raises(StoreError):
        store.start("../../scenario", "governed-v1", "launch-one")


# Construct matching pipeline metadata without copying the full private actor state.
def matching_evaluation(run):
    cfg = {"scenario": "scenario_01", "case_id": "example__scenario_01", "system": "aos_artifact_requests", "agent_llm": "test-model", "run_index": 0}
    pipeline = {"config": cfg, "final_response": run["final_response"],
                "task_description": run["projection"]["task"],
                "gathered_info": "\n".join(f"{c['text']} [evidence: {', '.join(c['evidence_refs'])}]" for r in run["responses"] for c in r["response"]["claims"]),
                "log": [{"request_id": e["request_id"], "kind": e["kind"], "message": e["body"],
                         "sender": "A" if e["sender_id"] == "agent-001" else "B",
                         "receiver": "A" if e["recipient_id"] == "agent-001" else "B"} for e in run["events"]]}
    judgment = {"config": {**cfg, "judge_llm": "judge-model", "judge_mode": "chain", "agent_audit": True},
                "utility": {"score": 1}, "completeness": {"n_covered": 1, "n_approp": 1, "rate": 1.0, "attributes": {"a1": {"ground_truth": SECRET}}},
                "V_A2A": {"n_flagged": 1, "n_inapp": 8, "rate": 0.125, "flagged": ["a2"], "raw": SECRET, "attributes": {"a2": {"explanation": SECRET}}}}
    return pipeline, judgment


# Display saved scores and flag IDs while omitting evaluator-only facts and prompts.
def test_saved_evaluation_is_separate_and_minimized(tmp_path):
    run = sample_run()
    pipeline, judgment = matching_evaluation(run)
    store = DemoStore(tmp_path / "demo")
    record = store.add_replay(save_json(tmp_path / "run.json", run),
                              pipeline=save_json(tmp_path / "pipeline.json", pipeline),
                              evaluation=save_json(tmp_path / "judgment.json", judgment))
    result = store.snapshot(record["run_id"])
    assert result["evaluation"]["binding"] == "operator_attached"
    assert result["evaluation"]["metrics"]["V_A2A"]["flagged"] == ["a2"]
    assert result["view"]["evaluation"]["status"] == "not_loaded"
    assert SECRET not in json.dumps(result)


# Refuse a score attachment whose export answer differs even when filenames look plausible.
def test_mismatched_evaluation_export_is_rejected(tmp_path):
    run = sample_run()
    pipeline, judgment = matching_evaluation(run)
    pipeline["final_response"] = "No."
    with pytest.raises(ValueError):
        DemoStore(tmp_path / "demo").add_replay(save_json(tmp_path / "run.json", run),
            pipeline=save_json(tmp_path / "pipeline.json", pipeline), evaluation=save_json(tmp_path / "judgment.json", judgment))


# Keep the last validated view if the next runtime snapshot is temporarily incomplete.
def test_bad_live_snapshot_retains_previous_view(tmp_path, monkeypatch):
    monkeypatch.setattr(DemoStore, "_launch", lambda *args: os.getpid())
    store = DemoStore(tmp_path / "demo")
    case = store.catalog()["cases"][0]["case_id"]
    record = store.start(case, "governed-v1", "launch-one")
    directory = store._directory(record["run_id"])
    (directory / "actor").mkdir()
    save_json(directory / "actor/run.json", sample_run())
    first = store.snapshot(record["run_id"])
    (directory / "actor/run.json").write_text('{"status":')
    second = store.snapshot(record["run_id"])
    assert first["view"] == second["view"]
    assert second["warnings"]


# Exercise the accepted detached worker using actual owner services and the scripted fixture only.
def test_offline_live_worker_finishes_and_persists(tmp_path):
    store = DemoStore(tmp_path / "demo")
    case = store.catalog()["cases"][0]["case_id"]
    receipt = store.start(case, "governed-v1", "launch-offline")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        row = store.runs()["runs"][0]
        if row["status"] not in {"queued", "running"}:
            break
        time.sleep(0.1)
    snapshot = store.snapshot(receipt["run_id"])
    assert snapshot["metadata"]["status"] == "completed", snapshot
    assert snapshot["view"]["a2a_records"] == 2
    assert "DEMO_PRIVATE" not in json.dumps(snapshot)
    assert snapshot["evaluation"]["status"] == "not_loaded"
    store.close()


# A contradictory protocol marker cannot smuggle unrecorded conflict behavior into a legacy replay.
def test_protocol_observation_consistency():
    run = sample_run()
    for field in ("conflicts", "conflict_audit", "conflict_detection"):
        changed = copy.deepcopy(run)
        changed[field] = []
        with pytest.raises(ReplayError):
            build_view(changed, "test-run", "a" * 64)
    run["decision_observation"] = {"schema": "aos-decision-observation-v1", "status": "not_started"}
    with pytest.raises(ReplayError):
        build_view(run, "test-run", "a" * 64)
    run.pop("decision_observation")
    run["config"]["protocol"] = "aos-artifact-requests-observed-v2"
    with pytest.raises(ReplayError):
        build_view(run, "test-run", "a" * 64)


# Nonfinite duration data cannot become invalid JSON or a misleading replay timing badge.
def test_nonfinite_duration_is_not_projected():
    run = sample_run()
    for value in (float("inf"), float("nan"), -1, True):
        run["pipeline_time"] = value
        assert build_view(run, "test-run", "a" * 64)["recorded_seconds"] is None


# Matching answer text alone cannot associate a different task, speaker, or gathered context.
@pytest.mark.parametrize("field", ["task_description", "gathered_info", "sender", "receiver"])
def test_evaluation_requires_matching_context_and_routing(tmp_path, field):
    run = sample_run()
    pipeline, judgment = matching_evaluation(run)
    if field in {"sender", "receiver"}:
        pipeline["log"][0][field] = "Wrong person"
    else:
        pipeline[field] = "Different input"
    with pytest.raises(ValueError):
        DemoStore(tmp_path / "demo").add_replay(save_json(tmp_path / "run.json", run),
            pipeline=save_json(tmp_path / "pipeline.json", pipeline), evaluation=save_json(tmp_path / "judgment.json", judgment))


# Use the official agent-context union denominator rather than an absent top-level count.
def test_agent_context_total_schema(tmp_path):
    run = sample_run()
    pipeline, judgment = matching_evaluation(run)
    judgment["V_A"] = {"total": {"n_flagged": 1, "n_potential": 19, "flagged": ["a3"], "rate": 1 / 19},
                       "Private owner name": {"attributes": {"secret": SECRET}}}
    store = DemoStore(tmp_path / "demo")
    record = store.add_replay(save_json(tmp_path / "run.json", run),
        pipeline=save_json(tmp_path / "pipeline.json", pipeline), evaluation=save_json(tmp_path / "judgment.json", judgment))
    evaluation = store.snapshot(record["run_id"])["evaluation"]
    assert evaluation["metrics"]["V_A"]["eligible"] == 19
    assert next(row["value"] for row in evaluation["rows"] if row["label"] == "Agent-context flags") == "1 / 19"
    assert SECRET not in json.dumps(evaluation)


# A version label must preserve both the decision protocol and owner review treatment.
def test_observed_protocol_cannot_claim_prompt_only():
    run = sample_run()
    run["config"].update(protocol="aos-artifact-requests-observed-v2", mode="prompt_only")
    run["decision_observation"] = {"schema": "aos-decision-observation-v1", "status": "not_started"}
    with pytest.raises(ReplayError):
        build_view(run, "test-run", "a" * 64)
