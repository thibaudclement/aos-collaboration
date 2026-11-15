# Verify the shipped catalogue, faithful replays, and no-key local workflow.

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from aos.collaboration.demo.bootstrap import bundled_data, seed_bundle
from aos.collaboration.demo.server import CollaborationServer
from aos.collaboration.demo.store import DemoStore, StoreError, write_private
from test_collaboration_demo_store import sample_run

JIRA08_RUN_ID = "run-ebf7ee121c8ac0d7a463f91a5eb5800e"
JIRA08_ACTOR_SHA256 = "ebf7ee121c8ac0d7a463f91a5eb5800e0dc87c40d49d8a5c39f6450e7944a322"
JIRA08_PIPELINE_SHA256 = "d84e8e9307e4bf6803c7829d93388fc0ca1af6ebf461f2e516373febf314e2e8"


# Seed every shipped case and replay without creating a model, then reopen the workspace.
def test_complete_bundle_is_portable_and_idempotent(tmp_path, monkeypatch):
    # Fail immediately if bootstrap or replay accidentally creates a provider client.
    def forbidden_client(*args, **kwargs):
        raise AssertionError("Replay must not construct a model client")
    monkeypatch.setattr("aos.llm.OpenAI", forbidden_client)
    root = tmp_path / "workspace"
    store = DemoStore(root)
    assert seed_bundle(store)["scenarios"] == 265
    catalog = store.catalog()
    assert len(catalog["cases"]) == 265
    assert catalog["versions"] == ["governed-v1", "observed-v2", "integrated-v3"]
    assert all(not item["live_enabled"] for item in catalog["cases"])
    records = store.runs()["runs"]
    assert len(records) == 109
    assert Counter(row["status"] for row in records) == {"completed": 105, "failed": 4}
    assert Counter(row["evaluation_status"] for row in records) == {"attached": 90, "not_loaded": 19}
    assert Counter(row["cohort"] for row in records)["external-baseline"] == 12
    assert Counter(row["cohort"] for row in records)["calibration"] == 18
    for record in records:
        snapshot = store.snapshot(record["run_id"])
        assert snapshot["view"]["run_id"] == record["run_id"]
        assert snapshot["view"]["scenario"]["case_id"] == record["case_id"]
        assert snapshot["evaluation"]["status"] == record["evaluation_status"]
    store.close()
    reopened = DemoStore(root)
    seed_bundle(reopened)
    assert reopened.runs()["runs"] == records
    assert len(reopened.catalog()["cases"]) == 265
    registry = json.loads((root / "registry.json").read_text())
    assert all(Path(item["scenario_path"]).is_relative_to(root) for item in registry["cases"].values())
    reopened.close()


# Bind every wave-two score to its recorded actor and retain the recovered evaluation's history.
def test_wave_two_receipt_bindings_and_recovered_actor_identity():
    bundle = bundled_data()
    manifest = json.loads((bundle / "manifest.json").read_text())
    items = [item for item in manifest["replays"] if item["cohort"] == "wave2"]
    cases = {"JIRA_allocation__scenario_08", "inpatient_discharge__scenario_02",
             "manuscript_submission__scenario_24", "meeting_allocation__scenario_19",
             "outgoing_museum_loan__scenario_18", "severity_classification__scenario_20",
             "special_event_permit_readiness__scenario_19", "thesis_readiness__scenario_15",
             "uas_flight_readiness__scenario_12"}
    arms = {"governed-v1": "governed-v1", "integrated-v3": "integrated", "native-decentralized": "native"}
    assert len(items) == 27
    assert {(item["case_id"], item["version"]) for item in items} == {(case, version) for case in cases for version in arms}
    seeds = {}
    for item in items:
        raw = (bundle / item["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        seed = json.loads(raw)
        seeds[item["run_id"]] = seed
        record, view, evaluation = seed["record"], seed["view"], seed["evaluation"]
        provenance = seed["provenance"].get("completed_wave2_source", seed["provenance"])
        assert record["run_id"] == view["run_id"] == item["run_id"]
        assert record["case_id"] == item["case_id"]
        assert record["version"] == item["version"]
        assert record["status"] == item["status"] == "completed"
        assert record["evaluation_status"] == item["evaluation_status"] == evaluation["status"] == "attached"
        assert record["source_sha256"] == view["source_sha256"] == provenance["source"]["sha256"]
        assert len(provenance["generation_receipts"]) == 1
        generation = provenance["generation_receipts"][0]
        judge = provenance["judge_receipt"]
        for receipt, stage in ((generation, "generate"), (judge, "judge")):
            assert receipt["status"] == "ok"
            assert receipt["stage"] == stage
            assert receipt["case"] == record["case_id"].replace("__", "/")
            assert receipt["arm"] == arms[record["version"]]
            assert len(receipt["sha256"]) == 64
            assert set(receipt["sha256"]) <= set("0123456789abcdef")
        assert record["generation_receipt_sha256"] == evaluation["generation_receipt_sha256"] == generation["sha256"]
        assert evaluation["stage_receipt_sha256"] == judge["sha256"]
        assert evaluation["stage_receipt_status"] == "ok"
        assert evaluation["binding"] == "verified_stage_receipt"
        assert evaluation["pipeline_content_checked"] is True
        assert evaluation["pipeline_sha256"] == provenance["pipeline"]["sha256"] == judge["evaluated_pipeline_sha256"]
        assert generation["verified_artifact_sha256"][provenance["pipeline"]["path"]] == evaluation["pipeline_sha256"]
        assert generation["verified_artifact_sha256"][provenance["source"]["path"]] == record["source_sha256"]
        assert evaluation["judgment_sha256"] == provenance["judgment"]["sha256"]
        assert judge["verified_artifact_sha256"][provenance["judgment"]["path"]] == evaluation["judgment_sha256"]
        if record["version"] == "native-decentralized":
            assert record["source_sha256"] == evaluation["pipeline_sha256"]
            assert "response_contract_id" not in record
            assert "response_contract_id" not in view
            assert "response contract" not in record["title"].lower()
        else:
            assert record["response_contract_id"] == view["response_contract_id"] == "aos-response-contract-v1"
    recovered = seeds[JIRA08_RUN_ID]
    assert recovered["record"]["source_sha256"] == JIRA08_ACTOR_SHA256
    assert recovered["evaluation"]["pipeline_sha256"] == JIRA08_PIPELINE_SHA256
    assert recovered["record"]["generation_receipt_sha256"] == "9c02e0b54921ca34f584141cd9da57f94ef836c606eeaf2fa558817894e44b01"
    provenance = recovered["provenance"]
    assert [receipt["sha256"] for receipt in provenance["judge_failures"]] == [
        "6ab57d10296e04b550db873f27f20ac90cb4078e0f0c89b128e160b30b90a597",
        "9003ed353bb8570510af7b658af12ac603e6ccf9e00d44f18a4c0b7844bd70ab",
        "cdce157134811bc8e7bebbfecd5fdc8cb69ec4e00f792be5624404113d8f172d"]
    assert all(receipt["status"] == "failed" for receipt in provenance["judge_failures"])
    recovery = provenance["recovery"]
    assert recovery["previous_evaluation_status"] == "failed"
    assert recovery["current_evaluation_status"] == "attached"
    assert recovery["generation_reused_unchanged"] is True
    assert recovery["recorded_actor_sha256"] == JIRA08_ACTOR_SHA256
    assert recovery["recorded_pipeline_sha256"] == JIRA08_PIPELINE_SHA256
    assert recovery["reused_responses"] == 402
    assert recovery["historical_attempted_calls"] == 403
    assert recovery["historical_unknown_cost_calls"] == 1
    assert recovery["new_unknown_cost_calls"] == 0


# Refresh a persisted older workspace without duplicating its recovered actor or changing local imports.
def test_bundle_refresh_preserves_existing_actor_and_unrelated_local_record(tmp_path, monkeypatch):
    # Fail if refreshing recorded evidence constructs a provider client.
    def forbidden_client(*args, **kwargs):
        raise AssertionError("Bundle refresh must not construct a model client")
    monkeypatch.setattr("aos.llm.OpenAI", forbidden_client)
    root = tmp_path / "workspace"
    store = DemoStore(root)
    seed_bundle(store)
    manifest = json.loads((bundled_data() / "manifest.json").read_text())
    new_ids = {item["run_id"] for item in manifest["replays"]
               if item["cohort"] == "wave2" and item["run_id"] != JIRA08_RUN_ID}
    assert len(new_ids) == 26
    recovered_directory = root / "runs" / JIRA08_RUN_ID
    provenance = json.loads((recovered_directory / "provenance.json").read_text())
    failed_attempts = copy.deepcopy(provenance["judge_failures"])
    for key in ("completed_wave2_source", "judgment", "judge_receipt", "recovery"):
        provenance.pop(key)
    write_private(recovered_directory / "provenance.json", provenance)
    write_private(recovered_directory / "evaluation.json", {
        "status": "failed", "error_type": "ProviderError",
        "message": "The recorded judge attempts failed; no official evaluation was completed for this conversation.",
        "attempts": failed_attempts})
    with store._locked() as registry:
        for run_id in new_ids:
            registry["runs"].pop(run_id)
            shutil.rmtree(root / "runs" / run_id)
        record = registry["runs"][JIRA08_RUN_ID]
        record["evaluation_status"] = "failed"
        record["warning"] = "Three recorded judge attempts failed; no official evaluation was completed."
        earlier_record = copy.deepcopy(record)
        registry["bundle"].update(replays=83, manifest_sha256="0" * 64)
    older_ids = {record["run_id"] for record in store.runs()["runs"]}
    assert len(older_ids) == 83
    earlier = store.snapshot(JIRA08_RUN_ID)
    assert earlier["evaluation"]["status"] == "failed"
    earlier_view = (recovered_directory / "view.json").read_bytes()
    source = tmp_path / "local-run.json"
    write_private(source, sample_run())
    local = store.add_replay(source, title="My local conversation")
    local_id = local["run_id"]
    local_snapshot = store.snapshot(local_id)
    local_directory = root / "runs" / local_id
    local_files = {path.name: path.read_bytes() for path in local_directory.iterdir() if path.is_file()}
    store.close()
    refreshed = DemoStore(root)
    assert seed_bundle(refreshed)["replays"] == 109
    records = refreshed.runs()["runs"]
    assert len(records) == 110
    assert {record["run_id"] for record in records} == older_ids | new_ids | {local_id}
    persisted = json.loads((root / "registry.json").read_text())["runs"]
    assert [run_id for run_id, record in persisted.items() if record["source_sha256"] == JIRA08_ACTOR_SHA256] == [JIRA08_RUN_ID]
    updated = refreshed.snapshot(JIRA08_RUN_ID)
    assert updated["metadata"]["evaluation_status"] == updated["evaluation"]["status"] == "attached"
    for key in ("run_id", "source_sha256", "generation_receipt_sha256", "created", "updated"):
        assert persisted[JIRA08_RUN_ID][key] == earlier_record[key]
    assert (recovered_directory / "view.json").read_bytes() == earlier_view
    updated_provenance = json.loads((recovered_directory / "provenance.json").read_text())
    assert updated_provenance["judge_failures"] == failed_attempts
    assert updated_provenance["recovery"]["reused_responses"] == 402
    assert refreshed.snapshot(local_id) == local_snapshot
    assert {path.name: path.read_bytes() for path in local_directory.iterdir() if path.is_file()} == local_files
    refreshed.close()
    reopened = DemoStore(root)
    seed_bundle(reopened)
    assert reopened.runs()["runs"] == records
    assert reopened.snapshot(JIRA08_RUN_ID) == updated
    assert json.loads((recovered_directory / "provenance.json").read_text()) == updated_provenance
    assert reopened.snapshot(local_id) == local_snapshot
    assert {path.name: path.read_bytes() for path in local_directory.iterdir() if path.is_file()} == local_files
    reopened.close()


# Reject altered packaged evidence before importing any paid conversation.
def test_corrupt_bundle_cannot_be_imported(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(bundled_data(), bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    path = bundle / manifest["replays"][0]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    store = DemoStore(tmp_path / "workspace")
    with pytest.raises(StoreError, match="integrity"):
        seed_bundle(store, bundle)
    assert store.runs()["runs"] == []
    store.close()


# Keep the catalog and every replay accessible while the backend rejects a no-key launch.
def test_populated_server_without_credentials(tmp_path):
    store = DemoStore(tmp_path / "workspace", allow_live=False)
    seed_bundle(store)
    server = CollaborationServer(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(server.url + "/api/collaboration/catalog") as response:
            catalog = json.load(response)
        assert len(catalog["cases"]) == 265
        with urlopen(server.url + "/api/collaboration/runs") as response:
            records = json.load(response)["runs"]
        assert len(records) == 109
        request = Request(server.url + "/api/collaboration/start", method="POST",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps({"case_id": catalog["cases"][0]["case_id"],
                                           "version": "integrated-v3", "launch_id": "offline-test-001"}).encode())
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
        assert len(store.runs()["runs"]) == 109
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        store.close()
