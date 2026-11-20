# Verify offline replication exports and preserve the demo's public replay boundary.

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from aos.collaboration.demo.store import DemoStore
from aos.collaboration.replication.exports import export_bundle, plan_hash
from test_collaboration_demo_store import SECRET, matching_evaluation, sample_run


# Write a synthetic canonical record and return the exact byte identity.
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Supply native pipeline fields without fabricating AOS owner-release messages.
def native_pipeline():
    return {"config": {"system": "decentralized", "scenario": "scenario_01", "agent_llm": "test-model", "run_index": 0},
            "task_description": "Can the room seat 20?", "final_response": "Yes.\nExact café answer.",
            "gathered_info": "The room seats 24.", "n_a2a": 2, "pipeline_time": 1.25,
            "degraded": False, "agent_contents": {"A": SECRET, "B": SECRET},
            "log": [{"type": "sent", "agent": "A", "from_": "A", "to": "B", "message": "How many seats?"},
                    {"type": "received", "agent": "B", "from_": "A", "to": "A", "message": "The room seats 24."},
                    {"type": "gathered", "agent": "A", "message": "The room seats 24."},
                    {"type": "final", "agent": "A", "message": "Yes.\nExact café answer."}],
            "ground_truth": {"private": SECRET}}


# Create all three system outputs with real-shaped receipts and no inference provider.
@pytest.fixture
def batch(tmp_path):
    plan = {"schema": "aos-replication-plan-v1", "phase": "wave3", "actor_model": "test-model",
            "created": "2026-11-20T00:00:00Z", "specification": {"fixed_batch": True},
            "source_sha256": {"synthetic.py": "f" * 64}, "inputs": [], "runs": [], "stages": []}
    status = {"schema": "aos-replication-status-v1", "phase": "wave3", "records": [], "provider_requests": 0}
    for arm in ("native", "governed-v1", "integrated"):
        run = {"case": "example/scenario_01", "arm": arm, "run_index": 0,
               "path": f"slots/example/scenario_01/{arm}/run0", "maximum_calls": 3,
               "model": "test-model", "response_contract": arm != "native"}
        plan["runs"].append(run)
        directory = tmp_path / run["path"]
        actor = sample_run()
        actor["config"].pop("credential")
        actor["config"].update(system_version="integrated-v3" if arm == "integrated" else "governed-v1",
                               response_contract={"id": "aos-response-contract-v1"})
        actor["final_response"] = "Yes.\nExact café answer."
        pipeline, judgment = matching_evaluation(actor)
        if arm == "native":
            pipeline = native_pipeline()
            judgment["config"].update(system="decentralized")
        else:
            save(directory / "actor/run.json", actor)
            save(directory / "actor/owners/agent-002/audit/req-one.json",
                 {"request_id": "req-one", "context": {"owner_id": "agent-002"}, "evidence": SECRET})
            save(directory / "actor/manifest.json", {"credential": "SHOULD_NOT_ENTER_ARCHIVE"})
            save(directory / "actor/owners/agent-002/config.json", {"token": "SHOULD_NOT_ENTER_ARCHIVE"})
        pipeline_hash = save(directory / "pipeline.json", pipeline)
        judgment_hash = save(directory / "judgment.json", judgment)
        for stage in ("generate", "judge"):
            identity = f"example/scenario_01:{arm}:0:{stage}"
            row = {**run, "stage": stage, "id": identity}
            plan["stages"].append(row)
            status["records"].append({"id": identity, "case": run["case"], "arm": arm, "run_index": 0,
                                      "stage": stage, "status": "ok", "model_calls": 1, "known_response_cost_usd": 0.01})
            save(directory / f"{stage}-attempt.json", {"stage": row, "started": "2026-11-20T00:00:00Z"})
            receipt = {"schema": "aos-replication-stage-result-v1", "case": run["case"], "arm": arm, "run_index": 0,
                       "stage": stage, "status": "ok", "finished": "2026-11-20T00:00:01Z",
                       "artifact_sha256": {run["path"] + "/pipeline.json": pipeline_hash}}
            if stage == "judge":
                receipt.update(evaluated_pipeline_sha256=pipeline_hash)
                receipt["artifact_sha256"][run["path"] + "/judgment.json"] = judgment_hash
            elif arm != "native":
                receipt["artifact_sha256"][run["path"] + "/actor/run.json"] = hashlib.sha256((directory / "actor/run.json").read_bytes()).hexdigest()
            save(directory / f"{stage}.json", receipt)
            subdirectory = "generation" if stage == "generate" else "judging"
            save(directory / subdirectory / "calls/001.json", {"status": "ok", "provider_cost_usd": 0.01})
    save(tmp_path / "plan.json", plan)
    status["plan_sha256"] = plan_hash(plan)
    (tmp_path / ".env").write_text("API_KEY=SHOULD_NOT_ENTER_ARCHIVE\n")
    return tmp_path, plan, status


# Export exact public answers and evaluations while keeping private evidence out of replay files.
def test_exports_preserve_three_systems_without_opening_gui(batch, monkeypatch):
    root, plan, status = batch

    # Reject any accidental GUI workspace construction during CLI export.
    def no_store(*args, **kwargs):
        raise AssertionError("Export must not open the GUI store")

    monkeypatch.setattr(DemoStore, "__init__", no_store)
    before = {path: path.read_bytes() for path in root.glob("slots/**/*") if path.is_file()}
    result = export_bundle(root, plan, status)
    assert result["provider_requests"] == 0 and result["gui_imported"] is False
    assert result["available_runs"] == result["planned_runs"] == 3
    index = json.loads((root / "replay.json").read_text())
    assert index["all_planned_runs_available"] and index["gui_imported"] is False
    versions = set()
    for row in index["runs"]:
        snapshot = json.loads((root / row["path"]).read_text())
        versions.add(snapshot["record"]["version"])
        assert snapshot["view"]["answer"]["text"] == "Yes.\nExact café answer."
        assert snapshot["evaluation"]["binding"] == "verified_stage_receipt"
        assert snapshot["evaluation"]["metrics"]["utility"]["score"] == 1
        assert SECRET not in json.dumps(snapshot)
        assert bool(snapshot["record"].get("response_contract_id")) == (row["system"] != "native")
    assert versions == {"native-decentralized", "governed-v1", "integrated-v3"}
    readout = json.loads((root / "readout.json").read_text())
    assert len(readout["records"]) == 3 and all("official_metrics" in row for row in readout["records"])
    assert SECRET in json.dumps(readout)
    assert all(path.read_bytes() == raw for path, raw in before.items())


# Index every archived byte while omitting credentials, service configuration and raw input directories.
def test_analysis_archive_is_selective_and_byte_indexed(batch):
    root, plan, status = batch
    export_bundle(root, plan, status)
    with zipfile.ZipFile(root / "analysis.zip") as archive:
        index = json.loads(archive.read("bundle-index.json"))
        assert index["source_sha256"] == plan["source_sha256"] and index["inputs"] == plan["inputs"]
        assert set(archive.namelist()) == {row["path"] for row in index["files"]} | {"bundle-index.json"}
        for row in index["files"]:
            raw = archive.read(row["path"])
            assert hashlib.sha256(raw).hexdigest() == row["sha256"] and len(raw) == row["size_bytes"]
            assert b"SHOULD_NOT_ENTER_ARCHIVE" not in raw
        assert not any(name.endswith("manifest.json") or name == ".env" for name in archive.namelist())
        assert any("/audit/" in name for name in archive.namelist())


# Keep failed judge evidence without attaching a successful score to its replay.
def test_failed_judgment_keeps_replay_without_metrics(batch):
    root, plan, status = batch
    status["records"][1]["status"] = "failed"
    export_bundle(root, plan, status)
    replay = json.loads((root / "replay.json").read_text())["runs"][0]
    assert replay["replay"] == "exported" and replay["evaluation"] == "not_loaded"
    row = json.loads((root / "readout.json").read_text())["records"][0]
    assert "judgment" in row and "official_metrics" not in row and "official_metrics_unavailable" in row


# Reject accepted scores when the saved pipeline no longer matches its evaluation receipt.
def test_mismatched_judgment_binding_never_attaches_metrics(batch):
    root, plan, status = batch
    path = root / plan["runs"][0]["path"] / "judge.json"
    receipt = json.loads(path.read_text())
    receipt["evaluated_pipeline_sha256"] = "0" * 64
    save(path, receipt)
    export_bundle(root, plan, status)
    replay = json.loads((root / "replay.json").read_text())
    assert replay["available_runs"] == 2 and replay["runs"][0]["replay"] == "failed"
    assert "official_metrics" not in json.loads((root / "readout.json").read_text())["records"][0]


# Reject changed plan identities before replacing any export summaries.
def test_export_checks_plan_identity(batch):
    root, plan, status = batch
    status["plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recorded execution plan"):
        export_bundle(root, plan, status)
    assert not (root / "status.json").exists()


# Refuse credential-bearing canonical records instead of silently publishing or rewriting them.
def test_export_rejects_credentials_in_canonical_metadata(batch):
    root, plan, status = batch
    path = root / plan["runs"][0]["path"] / "pipeline.json"
    pipeline = json.loads(path.read_text())
    pipeline["config"]["api_key"] = "SHOULD_NOT_ENTER_ARCHIVE"
    save(path, pipeline)
    with pytest.raises(ValueError, match="Credential-bearing"):
        export_bundle(root, plan, status)
    assert not (root / "analysis.zip").exists()
