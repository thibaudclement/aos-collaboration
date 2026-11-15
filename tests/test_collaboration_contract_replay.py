# Keep the response contract visible without publishing its private adapter audit.

from __future__ import annotations
import copy
import hashlib
import json
import pytest
from aos.collaboration.demo.store import DemoStore
from aos.collaboration.demo.view import ReplayError, build_view
from test_collaboration_demo_store import SECRET, sample_run, save_json

CONTRACT_ID = "aos-response-contract-v1"
PRIVATE_CONTRACT = "PRIVATE_CONTRACT_POLICY_AND_SIDECAR_DO_NOT_DISPLAY"


# Provide a canonical run whose serializer is separate from its owner treatment.
def contract_run(version="integrated-v3", status="completed"):
    run = sample_run()
    run["config"].update(system_version=version, response_contract={"id": CONTRACT_ID,
                                                                   "policy": PRIVATE_CONTRACT})
    run["decision_contract"] = {"id": CONTRACT_ID, "status": "applied" if status == "completed" else "failed",
                                "operation": PRIVATE_CONTRACT, "raw_sha256": "a" * 64,
                                "normalized_sha256": "b" * 64,
                                "sidecars": [PRIVATE_CONTRACT]}
    if status == "failed":
        run.update(status="failed", final_response="", used_evidence_refs=[])
        run["decision_audit"].append({"response": PRIVATE_CONTRACT})
    return run


# Retain each AOS identity while exposing only the known contract identifier.
@pytest.mark.parametrize("version", ["governed-v1", "integrated-v3"])
@pytest.mark.parametrize("status", ["completed", "failed"])
def test_contract_replay_retains_version_status_and_private_audit(tmp_path, version, status):
    run = contract_run(version, status)
    source = save_json(tmp_path / "run.json", run)
    before = source.read_bytes()
    store = DemoStore(tmp_path / "store")
    record = store.add_replay(source)
    snapshot = store.snapshot(record["run_id"])
    assert store.find_replay(hashlib.sha256(before).hexdigest())["run_id"] == record["run_id"]
    assert snapshot["metadata"]["version"] == version
    assert snapshot["metadata"]["status"] == status
    assert snapshot["metadata"]["response_contract_id"] == CONTRACT_ID
    assert snapshot["view"]["response_contract_id"] == CONTRACT_ID
    assert store.runs()["runs"][0]["response_contract_id"] == CONTRACT_ID
    assert snapshot["view"]["answer"]["text"] == run["final_response"]
    assert snapshot["view"]["answer"]["state"] == ("completed" if status == "completed" else "not_available")
    public = json.dumps(snapshot)
    assert SECRET not in public and PRIVATE_CONTRACT not in public
    assert "raw_sha256" not in public and "sidecars" not in public and "decision_contract" not in public
    assert source.read_bytes() == before
    store.close()


# Keep a failed generation authoritative even when an adapted answer was saved.
def test_contract_generation_failure_is_not_promoted_by_adapter_metadata(tmp_path):
    source = save_json(tmp_path / "run.json", contract_run())
    store = DemoStore(tmp_path / "store")
    record = store.add_replay(source, generation_receipt={"status": "failed", "sha256": "a" * 64})
    snapshot = store.snapshot(record["run_id"])
    assert snapshot["metadata"]["response_contract_id"] == CONTRACT_ID
    assert snapshot["metadata"]["status"] == snapshot["metadata"]["generation_stage_status"] == "failed"
    assert snapshot["view"]["answer"]["text"] == "Yes."
    assert snapshot["evaluation"]["status"] == "not_loaded"
    assert "without an accepted completion" in snapshot["warnings"][0]
    store.close()


# Reject unknown profiles rather than applying the tested contract's label.
@pytest.mark.parametrize("profile", [None, "aos-response-contract-v1", [], {}, {"id": None},
                                    {"id": "aos-response-contract-v99"}, {"id": [CONTRACT_ID]}])
def test_unknown_or_malformed_contract_profile_is_rejected(tmp_path, profile):
    run = contract_run()
    run["config"]["response_contract"] = profile
    with pytest.raises(ReplayError, match="Unsupported response contract profile"):
        build_view(run, "contract-test", "a" * 64)
    store = DemoStore(tmp_path / "store")
    with pytest.raises(ReplayError):
        store.add_replay(save_json(tmp_path / "run.json", run))
    assert store.runs()["runs"] == []
    store.close()


# Prevent the serializer from attaching to observed or prompt-only treatments.
@pytest.mark.parametrize("treatment", ["observed-v2", "prompt_only"])
def test_contract_profile_rejects_another_recorded_treatment(treatment):
    run = contract_run("governed-v1")
    if treatment == "observed-v2":
        run["config"].update(protocol="aos-artifact-requests-observed-v2", system_version=treatment)
        run["decision_observation"] = {"schema": "aos-decision-observation-v1", "status": "started"}
    else:
        run["config"]["mode"] = treatment
    with pytest.raises(ReplayError, match="Response contract does not match"):
        build_view(run, "contract-test", "a" * 64)


# Keep legacy records untagged even when a private audit names a contract.
def test_existing_records_are_not_retroactively_tagged(tmp_path):
    legacy = sample_run()
    legacy["decision_contract"] = {"id": CONTRACT_ID, "private": PRIVATE_CONTRACT}
    unchanged = copy.deepcopy(legacy)
    store = DemoStore(tmp_path / "store")
    record = store.add_replay(save_json(tmp_path / "run.json", legacy))
    snapshot = store.snapshot(record["run_id"])
    assert "response_contract_id" not in snapshot["metadata"]
    assert "response_contract_id" not in snapshot["view"]
    assert PRIVATE_CONTRACT not in json.dumps(snapshot)
    assert legacy == unchanged
    store.close()
