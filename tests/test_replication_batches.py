# Verify fixed experiment selections, source fidelity, and evaluator input separation.

import copy
import hashlib
import json
import pytest
from aos.collaboration.demo.bootstrap import bundled_data
from aos.collaboration.pisas import PisasError
from aos.collaboration.replication import batches, inputs


# Match the selected scientific slots to the bundled experiments without adding recovery attempts.
def test_fixed_batches_match_recorded_case_system_and_repeat_selections():
    bundle = bundled_data()
    manifest = json.loads((bundle / "manifest.json").read_text())
    expected_counts = {"external-baseline": 12, "calibration": 18, "wave1": 18, "wave2": 27, "wave3": 27}
    assert {row["name"]: row["runs"] for row in batches.list_batches()} == expected_counts
    for name, count in expected_counts.items():
        batch = batches.get_batch(name)
        runs = batches.planned_runs(batch)
        recorded = [json.loads((bundle / item["path"]).read_text())["record"]
                    for item in manifest["replays"] if item["cohort"] == name]
        identities = {(row["case_id"].replace("__", "/"), row["version"], row["run_index"])
                      for row in recorded}
        assert {(row["case"], row["version"], row["run_index"]) for row in runs} == identities
        assert len(runs) == len({row["path"] for row in runs}) == count
        assert all(row["response_contract"] == (name in {"wave2", "wave3"} and row["arm"] != "native")
                   for row in runs)
    assert len([item for item in manifest["replays"] if item["cohort"] == "wave1"]) == 19


# Retain the distinct Google baseline route and the Opus sampling compatibility settings.
def test_model_routes_and_sampling_profiles_remain_fixed():
    external = batches.planned_runs(batches.get_batch("external-baseline"))
    assert {(row["model"], row["base_url"], row["key_env"]) for row in external if row["arm"] == "governed-v1"} == {
        ("gemini-3.5-flash-lite", "https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY")}
    assert {(row["model"], row["base_url"], row["key_env"]) for row in external if row["arm"] == "native"} == {
        ("google/gemini-3.5-flash-lite", "https://openrouter.ai/api/v1", "API_KEY")}
    opus = batches.get_batch("wave3")
    assert opus["actor_model"] == "anthropic/claude-opus-5.5"
    assert opus["native_temperature"] is None
    assert opus["aos_temperature"] is None and opus["aos_max_tokens"] is None
    assert opus["judge_profile"] == "capacity" and opus["judge_timeout_seconds"] == 300
    assert batches.get_batch("calibration")["judge_profile"] == "legacy"
    changed = batches.get_batch("wave3")
    changed["actor_model"] = "replacement"
    with pytest.raises(ValueError, match="fixed"):
        batches.planned_runs(changed)
    assert batches.get_batch("wave3") == opus
    with pytest.raises(ValueError, match="Unknown batch"):
        batches.get_batch("canary")


# Reproduce all 195 prepared source files and keep scoring labels outside actor inputs.
def test_prepared_inputs_match_original_experiment_hashes(tmp_path):
    identities = {}
    cases = set()
    for item in batches.list_batches():
        batch = batches.get_batch(item["name"])
        root = tmp_path / item["name"]
        rows = batches.prepare_inputs(batch, root)
        for row in rows:
            case = row["case"]
            cases.add(case)
            actor = root / row["actor_scenario"]
            evaluator = root / row["judge_input"]
            assert set(path.name for path in actor.parent.iterdir()) == {"scenario.json"}
            assert actor.read_bytes() == (evaluator / "scenario.json").read_bytes()
            assert hashlib.sha256(actor.read_bytes()).hexdigest() == row["actor_sha256"]
            for name, expected in row["input_files"].items():
                raw = (evaluator / name).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == expected
                identities[case + "/" + name] = expected
        with pytest.raises(PisasError, match="already exist"):
            batches.prepare_inputs(batch, root)
    assert len(cases) == 39 and len(identities) == 195
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == "8dda2db8b4b239581096a91e33bf472c314c403f404b8a7ac0c03f8aab4a2cca"


# Refuse changed evaluator labels before creating any actor or evaluation input directory.
def test_corrupt_annotation_bundle_is_rejected_before_preparation(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    raw = (bundled_data() / "replication.json").read_bytes()
    (bundle / "replication.json").write_bytes(raw + b" ")
    monkeypatch.setattr(inputs, "bundled_data", lambda: bundle)
    target = tmp_path / "output"
    with pytest.raises(PisasError, match="integrity"):
        batches.prepare_inputs(batches.get_batch("wave3"), target)
    assert not target.exists()


# Supply an observed AOS decision with distinct owner and requester contexts.
def observed_run():
    return {"status": "completed", "config": {"model": "google/gemini-3.5-flash-lite", "mode": "governed",
                                                "run_index": 1, "top_k": 8},
            "projection": {"case_id": "example__scenario_01", "task_family": "example", "scenario_name": "scenario_01",
                           "task": "Choose a time.", "executor_id": "agent-001",
                           "participants": [{"id": "agent-001", "name": "Alex"}, {"id": "agent-002", "name": "Blair"}]},
            "final_response": "Meet at 10.", "gathered_info": "Blair is available at 10.",
            "decision_audit": [{"phase": "decide", "system": "Use shared evidence.", "prompt": "Blair is available at 10."}],
            "agent_contents": {"agent-001": "Blair is available at 10.", "agent-002": "Owner-only explanation."},
            "events": [{"kind": "response", "sender_id": "agent-002", "recipient_id": "agent-001",
                        "body": "Blair is available at 10.", "request_id": "req-one", "delivery": "confirmed"}],
            "n_a2a": 1, "pipeline_time": 1.5}


# Preserve the original evaluator format and distinguish owner contexts from released evidence.
def test_export_preserves_observed_messages_and_final_answer(tmp_path):
    run = observed_run()
    original = copy.deepcopy(run)
    target = batches.export_pisas(run, tmp_path)
    result = json.loads(target.read_text())
    assert target.name == "pipeline_scenario_01_run1.json"
    assert result["config"]["system"] == "aos_artifact_requests"
    assert result["config"]["run_index"] == 1
    assert result["final_response"] == run["final_response"]
    assert result["log"][0]["message"] == run["events"][0]["body"]
    assert result["log"][0]["sender"] == "Blair" and result["log"][0]["receiver"] == "Alex"
    assert result["agent_contents"]["Blair"] == "Owner-only explanation."
    assert "Owner-only" not in result["gathering_context_at_decision"]
    assert result["n_a2a"] == 1 and result["pipeline_time"] == 1.5
    assert run == original


# Prevent failed execution and unconfirmed delivery from entering a completed evaluator pipeline.
@pytest.mark.parametrize("change", [{"status": "failed"}, {"degraded": True}, {"n_a2a": 2}])
def test_export_rejects_incomplete_actor_records(tmp_path, change):
    run = observed_run()
    run.update(change)
    with pytest.raises(PisasError):
        batches.export_pisas(run, tmp_path)
    assert list(tmp_path.iterdir()) == []
