# Verify fixed-batch planning, retained paid attempts, and offline CLI exports.

from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from aos.collaboration.demo.store import DemoStore
from aos.collaboration.replication import runner
from aos.collaboration.replication.__main__ import main
from aos.collaboration.replication.provider import ProviderError, write_record
from test_collaboration_demo_store import matching_evaluation, sample_run
from test_replication_exports import native_pipeline


# Reject any provider construction or GUI opening during an offline operation.
def forbid_external_calls(monkeypatch):
    calls = []

    # Preserve evidence of an unexpected attempt before failing the offline check.
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Offline commands cannot construct a provider or GUI store")

    monkeypatch.setattr(runner.Transport, "__init__", forbidden)
    monkeypatch.setattr(runner, "credentials", forbidden)
    monkeypatch.setattr(DemoStore, "__init__", forbidden)
    return calls


# Prepare all real Wave 3 cases and common exports without reading credentials or constructing a provider.
def test_real_batch_preview_and_cli_are_offline(tmp_path, monkeypatch, capsys):
    calls = forbid_external_calls(monkeypatch)
    monkeypatch.setenv("API_KEY", "present-but-never-used")
    assert main(["run", "wave3", "--out", str(tmp_path)]) == 0
    output = capsys.readouterr()
    summary = json.loads(output.out)
    assert "OFFLINE PREVIEW" in output.err and "--execute" in output.err
    assert summary["preview"] and summary["runs"] == 27 and summary["stages"] == 54
    assert summary["model_calls"] == 0 and summary["statuses"] == {"not_started": 54}
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert len(plan["inputs"]) == 9 and len(plan["runs"]) == 27
    assert all((tmp_path / row["actor_scenario"]).is_file() for row in plan["inputs"])
    assert (tmp_path / "analysis.zip").is_file()
    assert calls == []


# Build synthetic canonical outputs while retaining the real planning, receipt and export code.
@pytest.fixture
def prepared(tmp_path, monkeypatch):
    state = SimpleNamespace(root=tmp_path, executions=[], transports=[], messages=[])
    monkeypatch.setattr(runner, "source_fingerprint", lambda batch: {"offline-test.py": "a" * 64})
    preview = runner.run("calibration", tmp_path, progress=state.messages.append)
    state.plan = json.loads((tmp_path / "plan.json").read_text())
    assert preview["status"]["stages"] == 36
    monkeypatch.setattr(runner, "credentials", lambda plan: {"API_KEY": "synthetic-only-key"})

    # Return an inert stage transport whose constructor records every attempted dispatch.
    def transport(*args, **kwargs):
        state.transports.append(kwargs)
        return SimpleNamespace(cost_required=True, timeout=kwargs["timeout"], base_url=kwargs["base_url"])

    # Preserve exactly one synthetic provider receipt for a selected stage.
    def call(root, stage, *, status="ok", cost=0.01):
        kind = "generation" if stage["stage"] == "generate" else "judging"
        write_record(root / stage["path"] / kind / "calls/001.json",
                     {"call": 1, "status": status, "provider_cost_usd": cost})

    # Exercise canonical output and binding generation without invoking an actor or judge model.
    def success(root, plan, stage, budget, progress):
        state.executions.append(stage["id"])
        directory = root / stage["path"]
        call(root, stage)
        if stage["stage"] == "judge":
            pipeline = json.loads((directory / "pipeline.json").read_text())
            judgment = {"config": {**pipeline["config"], "judge_llm": "synthetic-judge", "judge_mode": "chain"},
                        "utility": {"score": 1}, "completeness": {"n_covered": 1, "n_approp": 1, "rate": 1}}
            write_record(directory / "judgment.json", judgment)
            return {"health": {"degraded": False}, "evaluated_pipeline_sha256": runner.file_hash(directory / "pipeline.json")}
        if stage["arm"] == "native":
            pipeline = native_pipeline()
            pipeline["config"].update(scenario=stage["case"].split("/")[1], agent_llm=stage["model"],
                                      run_index=stage["run_index"])
        else:
            actor = sample_run()
            actor["config"].pop("credential")
            actor["config"].update(model=stage["model"], system_version=stage["version"], run_index=stage["run_index"])
            actor["projection"]["case_id"] = stage["case"].replace("/", "__")
            if stage["response_contract"]:
                actor["config"]["response_contract"] = {"id": "aos-response-contract-v1"}
            pipeline, _ = matching_evaluation(actor)
            pipeline["config"].update(case_id=actor["projection"]["case_id"], scenario=stage["case"].split("/")[1],
                                      agent_llm=stage["model"], run_index=stage["run_index"])
            write_record(directory / "actor/run.json", actor)
        write_record(directory / "pipeline.json", pipeline)
        return {"health": {"status": "completed", "degraded": False}, "usage": {"calls": 1, "cost_usd": 0.01}}

    state.transport, state.call, state.success = transport, call, success
    monkeypatch.setattr(runner, "execute_stage", success)
    return state


# Continue after a completed prefix and preserve every existing artifact without redispatching it.
def test_completed_prefix_continues_and_finished_batch_never_repeats(prepared):
    state = prepared
    first = state.plan["stages"][0]
    runner.attempt_stage(state.root, state.plan, first, object(), state.messages.append)
    previous = {path: path.read_bytes() for path in (state.root / first["path"]).rglob("*") if path.is_file()}
    result = runner.run("calibration", state.root, execute=True, progress=state.messages.append,
                        transport_factory=state.transport)
    assert result["status"]["statuses"] == {"ok": 36}
    assert result["status"]["model_calls"] == 36 and result["exports"]["available_runs"] == 18
    assert state.executions == [stage["id"] for stage in state.plan["stages"]]
    assert len(state.transports) == 35
    assert all(path.read_bytes() == raw for path, raw in previous.items())
    before = {path: path.read_bytes() for path in (state.root / "slots").rglob("*") if path.is_file()}
    repeated = runner.run("calibration", state.root, execute=True, progress=state.messages.append,
                          transport_factory=state.transport)
    assert repeated["executed"] is False and len(state.executions) == 36 and len(state.transports) == 35
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert "PAID EXECUTION" in "\n".join(state.messages)
    assert (state.root / "progress.log").is_file()


# Retain failed or interrupted paid attempts and block any implicit retry in that directory.
@pytest.mark.parametrize("failure", [ProviderError, KeyboardInterrupt])
def test_failed_or_interrupted_stage_is_preserved_without_retry(prepared, monkeypatch, failure):
    state = prepared

    # Record an uncertain dispatched call before the transport fails or the user interrupts it.
    def fail(root, plan, stage, budget, progress):
        state.executions.append(stage["id"])
        state.call(root, stage, status="started", cost=None)
        raise failure("Synthetic retained interruption")

    monkeypatch.setattr(runner, "execute_stage", fail)
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            runner.run("calibration", state.root, execute=True, progress=state.messages.append,
                       transport_factory=state.transport)
    else:
        result = runner.run("calibration", state.root, execute=True, progress=state.messages.append,
                            transport_factory=state.transport)
        assert result["status"]["stop"]
    status = runner.status("calibration", state.root)
    assert status["unknown_cost_calls"] == 1 and status["model_calls"] == 1 and status["next"] is None
    expected = "interrupted" if failure is KeyboardInterrupt else "failed"
    assert status["records"][0]["status"] == expected
    first = state.root / state.plan["stages"][0]["path"]
    before = {path: path.read_bytes() for path in first.rglob("*") if path.is_file()}
    resumed = runner.run("calibration", state.root, execute=True, progress=state.messages.append,
                         transport_factory=state.transport)
    assert resumed["executed"] is False and resumed["status"]["stop"]
    assert len(state.executions) == len(state.transports) == 1
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert (state.root / "readout.json").is_file() and (state.root / "analysis.zip").is_file()


# Detect changed frozen inputs before credentials are used or a provider transport is constructed.
@pytest.mark.parametrize("surface", ["actor", "judge"])
def test_changed_input_blocks_before_provider_dispatch(prepared, monkeypatch, surface):
    state = prepared
    selected = state.plan["inputs"][0]
    evaluator_file = next(name for name in selected["input_files"] if name not in {"scenario.json", "aos_source_manifest.json"})
    relative = selected["actor_scenario"] if surface == "actor" else selected["judge_input"] + "/" + evaluator_file
    path = state.root / relative
    path.write_bytes(path.read_bytes() + b" ")
    calls = forbid_external_calls(monkeypatch)
    with pytest.raises(runner.ReplicationError, match="changed"):
        runner.run("calibration", state.root, execute=True, progress=state.messages.append,
                   transport_factory=state.transport)
    assert calls == [] and state.executions == state.transports == []


# Stop later paid dispatches when a frozen input changes during an otherwise successful stage.
def test_midrun_input_change_preserves_completed_stage_without_more_calls(prepared, monkeypatch):
    state = prepared
    changed = state.root / state.plan["inputs"][1]["actor_scenario"]

    # Simulate an external edit after the first stage has produced its canonical output.
    def complete_then_edit(root, plan, stage, budget, progress):
        result = state.success(root, plan, stage, budget, progress)
        changed.write_bytes(changed.read_bytes() + b" ")
        return result

    monkeypatch.setattr(runner, "execute_stage", complete_then_edit)
    with pytest.raises(runner.ReplicationError, match="frozen actor scenario changed"):
        runner.run("calibration", state.root, execute=True, progress=state.messages.append,
                   transport_factory=state.transport)
    status = runner.status_from_plan(state.root, state.plan)
    assert status["records"][0]["status"] == "ok" and status["records"][1]["status"] == "not_started"
    assert status["model_calls"] == 1 and status["known_response_cost_usd"] == 0.01
    assert len(state.executions) == len(state.transports) == 1
    first = state.root / state.plan["stages"][0]["path"]
    assert (first / "generate.json").is_file() and not (first / "judge-attempt.json").exists()
    assert not (first / "judging/calls").exists()


# Preserve the original interrupt when rebuilding derived exports also encounters an I/O failure.
def test_export_failure_does_not_mask_paid_keyboard_interrupt(prepared, monkeypatch):
    state = prepared

    # Interrupt only after a dispatched call has its durable receipt.
    def interrupt(root, plan, stage, budget, progress):
        state.call(root, stage, status="started", cost=None)
        raise KeyboardInterrupt("Synthetic paid interruption")

    # Simulate a separate derived-file write failure while canonical attempts remain available.
    def failed_export(*args, **kwargs):
        raise OSError("Synthetic export failure")

    monkeypatch.setattr(runner, "execute_stage", interrupt)
    monkeypatch.setattr(runner, "export_bundle", failed_export)
    with pytest.raises(KeyboardInterrupt, match="paid interruption"):
        runner.run("calibration", state.root, execute=True, progress=state.messages.append,
                   transport_factory=state.transport)
    status = runner.status("calibration", state.root)
    assert status["records"][0]["status"] == "interrupted" and status["model_calls"] == 1
    assert any("Export did not finish" in message for message in state.messages)


# Keep status and export CLI actions offline while preserving every canonical attempt.
def test_status_and_export_cli_do_not_dispatch_models(prepared, monkeypatch, capsys):
    state = prepared
    calls = forbid_external_calls(monkeypatch)
    assert main(["status", "calibration", "--out", str(state.root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["model_calls"] == 0 and status["statuses"] == {"not_started": 36}
    assert main(["export", "calibration", "--out", str(state.root)]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["provider_requests"] == 0 and exported["gui_imported"] is False
    assert Path(exported["analysis"]).is_file() and calls == []


# Refuse a second active writer before it can construct a paid transport.
def test_batch_lock_prevents_concurrent_dispatch(prepared):
    state = prepared
    with runner.batch_lock(state.root):
        with pytest.raises(runner.ReplicationError, match="Another process"):
            runner.run("calibration", state.root, execute=True, transport_factory=state.transport)
    assert state.transports == []
