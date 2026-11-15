# Verify real service-to-coordinator handoffs and captured decision contexts.
# The scripted actors test orchestration and isolation mechanisms, not PiSAs
# reasoning performance or the semantic accuracy of an LLM privacy reviewer.

from __future__ import annotations
from contextlib import ExitStack
import json
import pytest
from aos.collaboration.experiment import prepare_experiment
from aos.collaboration import runtime as runtime_module
from aos.collaboration.runtime import RunFailed, run_experiment
from aos.collaboration.service import ArtifactService
from test_collaboration_service import (
    ALICE_SECRET, BLAIR_SECRET, PUBLIC_ALICE, PUBLIC_BLAIR,
    RecordingModel, config_path, owner_configs, projection,
)

# Start genuine HTTP endpoints while leaving service teardown deterministic.
def start_owners(root, stack, *, failed_owner=None, reject_all=False):
    models = {}
    for owner, (path, _) in owner_configs(root).items():
        model = RecordingModel(fail_phase="review" if owner == failed_owner else None,
                               reject_all=reject_all)
        service = ArtifactService(path, model=model)
        service.start()
        stack.callback(service.stop)
        models[owner] = model
    return models


# Confirm initialization creates separate owner workspaces with only their source data.
def test_prepare_preserves_initial_artifact_access_and_refuses_overwrite(tmp_path):
    case = projection()
    root = tmp_path / "experiment"
    prepare_experiment(case, root, model="fixture-model", top_k=None)
    owners = owner_configs(root)
    assert set(owners) == {"agent-001", "agent-002"}
    paths = set()
    for owner, (path, config) in owners.items():
        workspace = config_path(path, config, "workspace")
        paths.add(workspace)
        assert (workspace / "workspace.db").is_file()
        assert (workspace / ".git").is_dir()
        text = "\n".join((workspace / item["path"]).read_text() for item in config["artifacts"])
        expected = ALICE_SECRET if owner == "agent-001" else BLAIR_SECRET
        forbidden = BLAIR_SECRET if owner == "agent-001" else ALICE_SECRET
        assert expected in text
        assert forbidden not in text
        assert not config_path(path, config, "policy").is_relative_to(workspace)
        assert not config_path(path, config, "index").is_relative_to(workspace)
    assert len(paths) == 2
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        prepare_experiment(case, root, model="fixture-model", top_k=None)


# Require the final recipient to be exactly the authorized executor before any files exist.
@pytest.mark.parametrize("recipient", ["Blair", "alice", "Alice "])
def test_prepare_rejects_a_different_final_recipient_before_creating_root(tmp_path, recipient):
    case = projection()
    case["recipient"] = recipient
    root = tmp_path / "experiment"
    with pytest.raises(ValueError, match="(?i)recipient|executor"):
        prepare_experiment(case, root, model="fixture-model", top_k=None)
    assert not root.exists()


# Exercise real HTTP handoffs and require a fresh decision context of released evidence.
def test_run_uses_real_releases_and_a_fresh_decision_context(tmp_path):
    case = projection()
    root = tmp_path / "experiment"
    prepare_experiment(case, root, model="fixture-model", top_k=None)
    decision = RecordingModel()
    with ExitStack() as stack:
        models = start_owners(root, stack)
        run_experiment(root, external_services=True, model=decision)
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "completed"
    assert record.get("error") in {None, ""}
    assert PUBLIC_ALICE in record["gathered_info"]
    assert PUBLIC_BLAIR in record["gathered_info"]
    assert record["final_response"]
    assert len(decision.calls) == 1
    assert "PHASE: decide" in decision.calls[0]["system"]
    assert PUBLIC_ALICE in decision.calls[0]["prompt"]
    assert PUBLIC_BLAIR in decision.calls[0]["prompt"]
    for secret in (ALICE_SECRET, BLAIR_SECRET, "PRIVATE_REVIEW_REASON_SENTINEL"):
        assert secret not in decision.calls[0]["prompt"]
        assert secret not in record["gathered_info"]
        assert secret not in json.dumps(record["events"])
    assert ALICE_SECRET in json.dumps(models["agent-001"].calls)
    assert BLAIR_SECRET not in json.dumps(models["agent-001"].calls)
    assert BLAIR_SECRET in json.dumps(models["agent-002"].calls)
    assert ALICE_SECRET not in json.dumps(models["agent-002"].calls)
    assert BLAIR_SECRET not in record["agent_contents"]["agent-001"]
    assert ALICE_SECRET not in record["agent_contents"]["agent-002"]
    requests = [event for event in record["events"] if event["kind"] == "request"]
    responses = [event for event in record["events"] if event["kind"] == "response"]
    assert requests and responses
    assert all(event["sender_id"] != event["recipient_id"] for event in record["events"])
    assert all(event["delivery"] == "confirmed" for event in record["events"])
    assert record["n_a2a"] == len(record["events"]) == 2
    assert {event["recipient_id"] for event in requests} == {"agent-002"}


# Preserve an over-refusing system's empty result as a completed conversation.
def test_empty_releases_are_completed(tmp_path):
    root = tmp_path / "experiment"
    prepare_experiment(projection(), root, model="fixture-model", top_k=None)
    decision = RecordingModel()
    with ExitStack() as stack:
        start_owners(root, stack, reject_all=True)
        run_experiment(root, external_services=True, model=decision)
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["insufficient_evidence"] is True
    assert record["gathered_info"] == ""
    assert all(item["response"]["status"] == "unavailable" for item in record["responses"])


# Record a model failure as a failed run without calling the final decision model.
def test_run_failure_is_persisted(tmp_path):
    root = tmp_path / "experiment"
    prepare_experiment(projection(), root, model="fixture-model", top_k=None)
    decision = RecordingModel()
    with ExitStack() as stack:
        start_owners(root, stack, failed_owner="agent-002")
        with pytest.raises(RunFailed):
            run_experiment(root, external_services=True, model=decision)
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "failed"
    assert record.get("error")
    assert decision.calls == []


# Preserve uncertain delivery when transport and private receipt collection both fail.
def test_transport_failure_records_attempted_delivery_without_inventing_receipt(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    manifest = prepare_experiment(projection(), root, model="fixture-model", top_k=None)
    peer = next(owner for owner in manifest["owners"] if owner["id"] != manifest["executor_id"])
    decision = RecordingModel()
    attempts = []
    original_request = runtime_module.request_json
    with ExitStack() as stack:
        models = start_owners(root, stack)
        peer_url = json.loads((root / peer["endpoint"]).read_text())["url"]

        # Simulate a lost connection without letting an audit falsely confirm acceptance.
        def unavailable_peer(url, token, data=None, *, timeout=210):
            if url.startswith(peer_url + "/") and (url.endswith("/request") or "/audit?" in url):
                attempts.append(url)
                raise OSError("Simulated unavailable transport and receipt channel")
            return original_request(url, token, data, timeout=timeout)

        monkeypatch.setattr(runtime_module, "request_json", unavailable_peer)
        with pytest.raises(RunFailed):
            run_experiment(root, external_services=True, model=decision)
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["degraded"] is True
    assert len(attempts) == 2
    assert record["n_a2a"] == 1
    assert len(record["events"]) == 1
    event = record["events"][0]
    assert event["kind"] == "request"
    assert event["sender_id"] == manifest["executor_id"]
    assert event["recipient_id"] == peer["id"]
    assert event["delivery"] == "attempted"
    assert models[peer["id"]].calls == []
    assert decision.calls == []


# A prompt-only comparator must retain actual disclosures instead of filtering its trace.
def test_prompt_only_baseline_exposes_and_records_its_unguarded_claims(tmp_path):
    root = tmp_path / "experiment"
    prepare_experiment(projection(), root, model="fixture-model", mode="prompt_only", top_k=None)
    decision = RecordingModel()
    with ExitStack() as stack:
        models = start_owners(root, stack)
        run_experiment(root, external_services=True, model=decision)
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "completed"
    assert BLAIR_SECRET in record["gathered_info"]
    assert BLAIR_SECRET in json.dumps(record["events"])
    assert BLAIR_SECRET in record["agent_contents"]["agent-001"]
    assert BLAIR_SECRET in decision.calls[0]["prompt"]
    assert all("PHASE: review" not in call["system"]
               for model in models.values() for call in model.calls)


# Missing owner services must fail before any final decision is fabricated.
def test_external_service_absence_is_an_explicit_run_failure(tmp_path):
    root = tmp_path / "experiment"
    prepare_experiment(projection(), root, model="fixture-model", top_k=None)
    decision = RecordingModel()
    with pytest.raises(RunFailed):
        run_experiment(root, external_services=True, model=decision)
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "failed"
    assert decision.calls == []
