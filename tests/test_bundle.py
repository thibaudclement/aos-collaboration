# Verify the shipped catalogue, faithful replays, and no-key local workflow.

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from aos.collaboration.demo.bootstrap import bundled_data, seed_bundle
from aos.collaboration.demo.server import CollaborationServer
from aos.collaboration.demo.store import DemoStore, StoreError, write_private
from aos.collaboration.pisas import load_scenario
from test_collaboration_demo_store import sample_run

JIRA08_RUN_ID = "run-ebf7ee121c8ac0d7a463f91a5eb5800e"
JIRA08_ACTOR_SHA256 = "ebf7ee121c8ac0d7a463f91a5eb5800e0dc87c40d49d8a5c39f6450e7944a322"
JIRA08_PIPELINE_SHA256 = "d84e8e9307e4bf6803c7829d93388fc0ca1af6ebf461f2e516373febf314e2e8"
JIRA11_RUN_ID = "run-67a113b254558e2facc62cd64680835c"
JIRA11_ACTOR_SHA256 = "67a113b254558e2facc62cd64680835ce747258102595415195250ceb8266269"
JIRA11_PIPELINE_SHA256 = "64951920c330dfad917b6bebe3bc8702c57154692fcbc03e9d65bd3cf05cc181"


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
    assert len(records) == 136
    assert Counter(row["status"] for row in records) == {"completed": 132, "failed": 4}
    assert Counter(row["evaluation_status"] for row in records) == {"attached": 117, "not_loaded": 19}
    assert Counter(row["cohort"] for row in records)["external-baseline"] == 12
    assert Counter(row["cohort"] for row in records)["calibration"] == 18
    references = json.loads((bundled_data() / "benchmark_answers.json").read_text())["cases"]
    cases = {item["case_id"]: item for item in catalog["cases"]}
    for record in records:
        snapshot = store.snapshot(record["run_id"])
        assert snapshot["view"]["run_id"] == record["run_id"]
        assert snapshot["view"]["scenario"]["case_id"] == record["case_id"]
        assert snapshot["evaluation"]["status"] == record["evaluation_status"]
        reference = snapshot["benchmark_reference"]
        assert reference["status"] == "available"
        assert reference["case_id"] == record["case_id"]
        assert reference["answer"] == references[record["case_id"]]["answer"]
        assert reference["scenario_sha256"] == cases[record["case_id"]]["source_sha256"]
        assert "benchmark_reference" not in snapshot["view"]
        directory = root / "runs" / record["run_id"]
        assert snapshot["view"] == json.loads((directory / "view.json").read_text())
        assert snapshot["evaluation"] == json.loads((directory / "evaluation.json").read_text())
    store.close()
    reopened = DemoStore(root)
    seed_bundle(reopened)
    assert reopened.runs()["runs"] == records
    assert len(reopened.catalog()["cases"]) == 265
    registry = json.loads((root / "registry.json").read_text())
    assert all(Path(item["scenario_path"]).is_relative_to(root) for item in registry["cases"].values())
    assert set(registry["benchmark_references"]) == set(cases)
    reopened.close()


# Preserve explicit labels and complete allocation outputs from the pinned benchmark annotations.
def test_benchmark_answers_cover_every_bundled_scenario():
    bundle = bundled_data()
    manifest = json.loads((bundle / "manifest.json").read_text())
    raw = (bundle / manifest["benchmark_answers"]["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["benchmark_answers"]["sha256"]
    references = json.loads(raw)
    assert references["schema"] == "aos-benchmark-answers-v1"
    assert references["dataset"] == manifest["dataset"]
    cases = references["cases"]
    assert set(cases) == {item["case_id"] for item in manifest["scenarios"]}
    kinds = Counter()
    for item in manifest["scenarios"]:
        reference = cases[item["case_id"]]
        assert set(reference) == {"scenario_sha256", "utility_sha256", "answer"}
        assert reference["scenario_sha256"] == item["sha256"]
        assert len(reference["utility_sha256"]) == 64
        assert set(reference["utility_sha256"]) <= set("0123456789abcdef")
        answer = reference["answer"]
        kind = "decision" if isinstance(answer, str) else next(iter(answer))
        kinds[kind] += 1
        if kind == "decision":
            assert answer.strip() and answer not in {"correct_assignment", "correct_schedule"}
        else:
            assert set(answer) == {kind} and isinstance(answer[kind], dict) and answer[kind]
    assert kinds == {"decision": 205, "assignment": 25, "schedule": 35}
    assert cases["uas_flight_readiness__scenario_20"]["answer"] == "resolve_altitude"
    assert cases["thesis_readiness__scenario_17"]["answer"] == "ready_with_conditions"
    replication = json.loads((bundle / "replication.json").read_text())
    for name, case in replication["cases"].items():
        annotation = case["annotations"]["utility.json"]
        utility = json.loads(annotation["text"])
        answer = utility["oracle_answer"]
        if "oracle_assignment" in utility:
            answer = {"assignment": utility["oracle_assignment"]}
        elif "oracle_schedule" in utility:
            answer = {"schedule": utility["oracle_schedule"]}
        reference = cases[name.replace("/", "__")]
        assert reference["utility_sha256"] == hashlib.sha256(annotation["text"].encode()).hexdigest()
        assert reference["answer"] == answer


# Keep benchmark information outside actor inputs and bind live references to frozen launch settings.
def test_queued_benchmark_reference_is_operator_only(tmp_path, monkeypatch):
    monkeypatch.setattr(DemoStore, "_launch", lambda *args: os.getpid())
    store = DemoStore(tmp_path / "workspace", allow_live=True)
    seed_bundle(store)
    case_id = "uas_flight_readiness__scenario_20"
    canary = "BENCHMARK_ANSWER_OPERATOR_ONLY_CANARY"
    with store._locked() as registry:
        registry["benchmark_references"][case_id]["answer"] = canary
        registered = copy.deepcopy(registry["cases"][case_id])
    receipt = store.start(case_id, "integrated-v3", "benchmark-reference-live")
    snapshot = store.snapshot(receipt["run_id"])
    assert snapshot["metadata"]["status"] == "queued"
    assert snapshot["view"] is None
    assert snapshot["benchmark_reference"]["answer"] == canary
    assert snapshot["evaluation"] == {"status": "not_loaded"}
    assert canary not in json.dumps(store.catalog())
    assert canary not in json.dumps(load_scenario(registered["scenario_path"]))
    with store._locked() as registry:
        assert canary not in json.dumps(registry["runs"][receipt["run_id"]])
        registry["cases"][case_id]["source_sha256"] = "0" * 64
    assert store.snapshot(receipt["run_id"])["benchmark_reference"]["answer"] == canary
    with store._locked() as registry:
        registry["cases"][case_id] = registered
        registry["runs"][receipt["run_id"]]["case"]["source_sha256"] = "0" * 64
    assert store.snapshot(receipt["run_id"])["benchmark_reference"] == {"status": "not_available"}
    store.close()


# Omit unavailable or mismatched references without changing the saved outcome or evaluation.
def test_benchmark_reference_fallback_preserves_replay(tmp_path):
    store = DemoStore(tmp_path / "workspace")
    seed_bundle(store)
    before = store.snapshot(JIRA11_RUN_ID)
    case_id = before["metadata"]["case_id"]
    directory = store._directory(JIRA11_RUN_ID)
    original_files = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    with store._locked() as registry:
        reference = registry["benchmark_references"].pop(case_id)
        scenario_hash = registry["cases"][case_id]["source_sha256"]
    missing = store.snapshot(JIRA11_RUN_ID)
    assert missing["benchmark_reference"] == {"status": "not_available"}
    assert missing["view"] == before["view"] and missing["evaluation"] == before["evaluation"]
    with store._locked() as registry:
        registry["benchmark_references"][case_id] = reference
        registry["cases"][case_id]["source_sha256"] = "0" * 64
    mismatch = store.snapshot(JIRA11_RUN_ID)
    assert mismatch["benchmark_reference"] == {"status": "not_available"}
    assert mismatch["view"] == before["view"] and mismatch["evaluation"] == before["evaluation"]
    with store._locked() as registry:
        registry["cases"][case_id]["source_sha256"] = scenario_hash
    changed = copy.deepcopy(before["view"])
    changed["scenario"]["case_id"] = "uas_flight_readiness__scenario_20"
    write_private(directory / "view.json", changed)
    assert store.snapshot(JIRA11_RUN_ID)["benchmark_reference"] == {"status": "not_available"}
    (directory / "view.json").write_bytes(original_files["view.json"])
    assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == original_files
    source = tmp_path / "local-run.json"
    write_private(source, sample_run())
    imported = store.add_replay(source)
    unknown = store.snapshot(imported["run_id"])
    assert unknown["benchmark_reference"] == {"status": "not_available"}
    assert unknown["view"]["answer"]["text"] == "Yes."
    store.close()


# Reject a rehashed answer bundle that claims a different scenario revision.
def test_benchmark_answers_require_matching_scenario_hashes(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(bundled_data(), bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    answers_path = bundle / manifest["benchmark_answers"]["path"]
    answers = json.loads(answers_path.read_text())
    answers["cases"]["uas_flight_readiness__scenario_20"]["scenario_sha256"] = "0" * 64
    write_private(answers_path, answers)
    manifest["benchmark_answers"]["sha256"] = hashlib.sha256(answers_path.read_bytes()).hexdigest()
    write_private(manifest_path, manifest)
    store = DemoStore(tmp_path / "workspace")
    with pytest.raises(StoreError):
        seed_bundle(store, bundle)
    assert store.runs()["runs"] == []
    store.close()


# Preserve the bytes and paths of all prior scenarios and recorded conversations.
def test_prior_bundle_inputs_remain_byte_identical():
    bundle = bundled_data()
    manifest = json.loads((bundle / "manifest.json").read_text())
    prior_replays = [item for item in manifest["replays"] if item["cohort"] != "wave3"]
    assert len(prior_replays) == 109
    assert len(manifest["scenarios"]) == 265
    identities = {item["path"]: hashlib.sha256((bundle / item["path"]).read_bytes()).hexdigest()
                  for item in manifest["scenarios"] + prior_replays}
    assert len(identities) == 374
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == "87d9a65ca63f24e34f93ea1638af362f26935981939bd6ac24bbf2a11c9c765d"


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


# Bind all Opus replays to the completed actor and judge receipts for their exact case and system.
def test_opus_wave_receipt_bindings_and_system_identity():
    bundle = bundled_data()
    manifest = json.loads((bundle / "manifest.json").read_text())
    items = [item for item in manifest["replays"] if item["cohort"] == "wave3"]
    cases = {"JIRA_allocation__scenario_11", "inpatient_discharge__scenario_16",
             "manuscript_submission__scenario_02", "meeting_allocation__scenario_25",
             "outgoing_museum_loan__scenario_08", "severity_classification__scenario_23",
             "special_event_permit_readiness__scenario_15", "thesis_readiness__scenario_17",
             "uas_flight_readiness__scenario_20"}
    arms = {"governed-v1": "governed-v1", "integrated-v3": "integrated", "native-decentralized": "native"}
    assert len(items) == 27
    assert {(item["case_id"], item["version"]) for item in items} == {(case, version) for case in cases for version in arms}
    seeds = {}
    for item in items:
        raw = (bundle / item["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        seed = json.loads(raw)
        seeds[item["run_id"]] = seed
        record, view, evaluation, provenance = seed["record"], seed["view"], seed["evaluation"], seed["provenance"]
        assert record["run_id"] == view["run_id"] == item["run_id"]
        assert record["case_id"] == item["case_id"] == view["scenario"]["case_id"]
        assert record["version"] == item["version"]
        assert record["model"] == "anthropic/claude-opus-5.5"
        assert record["title"].startswith("Reserved wave 3 · Opus")
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
        assert record["generation_stage_status"] == evaluation["stage_receipt_status"] == "ok"
        assert record["generation_receipt_sha256"] == evaluation["generation_receipt_sha256"] == generation["sha256"]
        assert evaluation["stage_receipt_sha256"] == judge["sha256"]
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
    recovered = seeds[JIRA11_RUN_ID]
    assert recovered["record"]["source_sha256"] == JIRA11_ACTOR_SHA256
    assert recovered["evaluation"]["pipeline_sha256"] == JIRA11_PIPELINE_SHA256
    assert recovered["evaluation"]["stage_receipt_sha256"] == "5fec084347823446c481fcbfbb70e6c5763703420c1600ca9c0c7c86e6b87d51"
    provenance = recovered["provenance"]
    assert [receipt["sha256"] for receipt in provenance["judge_failures"]] == [
        "f19e4f1544dcfbef4875cda9b6029d880ad8ba2108397e711aec4c266c0bbb4a"]
    assert provenance["judge_failures"][0]["status"] == "failed"
    recovery = provenance["recovery"]
    assert recovery["previous_evaluation_status"] == "failed"
    assert recovery["current_evaluation_status"] == "attached"
    assert recovery["generation_reused_unchanged"] is True
    assert recovery["recorded_actor_sha256"] == JIRA11_ACTOR_SHA256
    assert recovery["recorded_pipeline_sha256"] == JIRA11_PIPELINE_SHA256
    assert recovery["reused_responses"] == 115
    assert recovery["new_attempted_calls"] == 612
    assert recovery["logical_calls"] == 727
    assert recovery["historical_attempted_calls"] == 116
    assert recovery["historical_unknown_cost_calls"] == 1
    assert recovery["new_unknown_cost_calls"] == 0
    assert all(warning in recovered["record"]["warning"] for warning in recovered["warnings"])
    assert any("unknown cost" in warning for warning in recovered["warnings"])


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
    opus_ids = {item["run_id"] for item in manifest["replays"] if item["cohort"] == "wave3"}
    assert len(opus_ids) == 27
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
        for run_id in new_ids | opus_ids:
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
    assert seed_bundle(refreshed)["replays"] == 136
    records = refreshed.runs()["runs"]
    assert len(records) == 137
    assert {record["run_id"] for record in records} == older_ids | new_ids | opus_ids | {local_id}
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


# Add the Opus collection to a saved 109-record workspace without changing its existing records.
def test_opus_bundle_refresh_preserves_prior_records_and_local_run(tmp_path, monkeypatch):
    # Fail if importing recorded conversations attempts to create a provider client.
    def forbidden_client(*args, **kwargs):
        raise AssertionError("Bundle refresh must not construct a model client")
    monkeypatch.setattr("aos.llm.OpenAI", forbidden_client)
    root = tmp_path / "workspace"
    store = DemoStore(root)
    seed_bundle(store)
    manifest = json.loads((bundled_data() / "manifest.json").read_text())
    opus_ids = {item["run_id"] for item in manifest["replays"] if item["cohort"] == "wave3"}
    assert len(opus_ids) == 27
    with store._locked() as registry:
        for run_id in opus_ids:
            registry["runs"].pop(run_id)
            shutil.rmtree(root / "runs" / run_id)
        registry["bundle"].update(replays=109, manifest_sha256="0" * 64)
    assert len(store.runs()["runs"]) == 109
    source = tmp_path / "local-run.json"
    write_private(source, sample_run())
    local_id = store.add_replay(source, title="My local conversation")["run_id"]
    local_snapshot = store.snapshot(local_id)
    prior_records = json.loads((root / "registry.json").read_text())["runs"]
    prior_files = {path.relative_to(root): path.read_bytes() for path in root.rglob("*")
                   if path.is_file() and path.name not in {"registry.json", "registry.lock"}}
    assert len(prior_records) == 110
    store.close()
    refreshed = DemoStore(root)
    assert seed_bundle(refreshed)["replays"] == 136
    records = refreshed.runs()["runs"]
    assert len(records) == 137
    registry = json.loads((root / "registry.json").read_text())
    assert set(registry["runs"]) == set(prior_records) | opus_ids
    assert {run_id: registry["runs"][run_id] for run_id in prior_records} == prior_records
    assert all((root / path).read_bytes() == raw for path, raw in prior_files.items())
    assert refreshed.snapshot(local_id) == local_snapshot
    current_files = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    refreshed.close()
    reopened = DemoStore(root)
    seed_bundle(reopened)
    assert reopened.runs()["runs"] == records
    assert reopened.snapshot(local_id) == local_snapshot
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == current_files
    reopened.close()


# Reject altered packaged evidence before importing any paid conversation.
@pytest.mark.parametrize("kind", ["replay", "benchmark_answers"])
def test_corrupt_bundle_cannot_be_imported(tmp_path, kind):
    bundle = tmp_path / "bundle"
    shutil.copytree(bundled_data(), bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    item = manifest["replays"][0] if kind == "replay" else manifest["benchmark_answers"]
    path = bundle / item["path"]
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
        assert len(records) == 136
        request = Request(server.url + "/api/collaboration/start", method="POST",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps({"case_id": catalog["cases"][0]["case_id"],
                                           "version": "integrated-v3", "launch_id": "offline-test-001"}).encode())
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
        assert len(store.runs()["runs"]) == 136
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        store.close()
