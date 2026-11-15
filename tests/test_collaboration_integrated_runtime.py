# Verify real integrated exchanges, preserved prompts, private lineage, and failed-attempt accounting.

from __future__ import annotations

from copy import deepcopy
import json
import threading

import pytest

from aos.collaboration.common import read_json
from aos.collaboration.integrated.components import extraction_system
from aos.collaboration.experiment import prepare_experiment
from aos.collaboration.integrated import runtime
from aos.collaboration.model import DECIDE, EXTRACT, REVIEW
from aos.collaboration.integrated.components import NORMALIZE
from aos.collaboration.runtime import RunFailed
from aos.tasks import Task
from aos.workspace import Workspace
from test_collaboration_service import ALICE_SECRET, BLAIR_SECRET, PUBLIC_ALICE, PUBLIC_BLAIR, projection


# Script mechanics without pretending to evaluate model reasoning or semantic privacy.
class IntegratedModel:
    # Select a specific malformed response or mutation while recording every call.
    def __init__(self, *, invalid_normalization=False, omit_all=False, before_review=None):
        self.calls = []
        self.invalid_normalization = invalid_normalization
        self.omit_all = omit_all
        self.before_review = before_review
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "reasoning_tokens": 0, "cost_usd": 0.0, "unpriced_calls": 0}

    # Return controlled judgments and account for exactly one invocation each time.
    def complete_json(self, prompt, system=None):
        payload = json.loads(prompt)
        phase = system.splitlines()[0].removeprefix("PHASE: ")
        self.calls.append({"phase": phase, "prompt": prompt, "system": system,
                           "payload": deepcopy(payload)})
        self.usage["calls"] += 1
        self.usage["cost_usd"] += 0.001
        if phase == "extract":
            return {"claims": [{"text": unit["text"], "evidence_ids": [unit["evidence_id"]]}
                               for unit in payload["evidence"]]}
        if phase == "normalize_release":
            return {"revisions": [{
                "parent_claim_id": "unknown-parent" if self.invalid_normalization else claim["claim_id"],
                "claims": [] if self.omit_all else [{"text": claim["text"].splitlines()[0],
                                                     "evidence_ids": claim["evidence_ids"]}],
                "reason": "PRIVATE_NORMALIZER_REASON_SENTINEL",
            } for claim in payload["proposal"]["claims"]]}
        if phase == "review":
            if self.before_review:
                self.before_review(payload)
            return {"decisions": [{"claim_id": claim["claim_id"], "supported": True,
                                    "appropriate": True, "permitted": True,
                                    "reason": "PRIVATE_REVIEW_REASON_SENTINEL"}
                                   for claim in payload["proposal"]["claims"]]}
        if phase == "decide":
            return {"final_response": "Meet Tuesday at 10:00." if payload["gathered_info"]
                    else "No meeting time established.", "used_evidence_refs": payload["evidence_refs"],
                    "insufficient_evidence": not bool(payload["gathered_info"])}
        raise AssertionError("Unexpected model phase")


# Prepare inferred-policy actor workspaces using the unchanged materializer.
def prepared(tmp_path, *, treatment="integrated", case=None):
    root = tmp_path / "actor"
    prepare_experiment(case or projection(), root, model="synthetic-integrated-test",
                       top_k=None if treatment == "integrated" else 8)
    return root


# Read one owner's terminal private request audit without relying on random request IDs.
def owner_audit(root, owner="agent-001"):
    paths = list((root / "owners" / owner / "audit").glob("*.json"))
    assert len(paths) == 1
    return read_json(paths[0])


# Exercise actual HTTP release, fresh decision context, lineage, and native AOS task persistence.
def test_integrated_run_preserves_prompts_and_private_boundaries(tmp_path):
    root = prepared(tmp_path)
    model = IntegratedModel()
    result = runtime.run(root, model)
    assert result == read_json(root / "run.json")
    assert result["status"] == "completed"
    assert result["config"]["system_version"] == "integrated-v3"
    assert result["config"]["protocol"] == "aos-artifact-requests-v1"
    assert result["usage"]["calls"] == 7
    assert result["n_a2a"] == len(result["events"]) == 2
    assert all(event["delivery"] == "confirmed" for event in result["events"])
    assert [call["phase"] for call in model.calls] == ["extract", "normalize_release", "review"] * 2 + ["decide"]
    expected = {"extract": extraction_system("cap24"), "normalize_release": NORMALIZE,
                "review": REVIEW, "decide": DECIDE}
    assert all(call["system"] == expected[call["phase"]] for call in model.calls)
    public = json.dumps({key: result[key] for key in
                         ("responses", "events", "gathered_info", "final_response", "decision_audit")})
    for secret in (ALICE_SECRET, BLAIR_SECRET, "PRIVATE_NORMALIZER_REASON_SENTINEL", "PRIVATE_REVIEW_REASON_SENTINEL"):
        assert secret not in public
    assert PUBLIC_ALICE in result["gathered_info"] and PUBLIC_BLAIR in result["gathered_info"]
    assert BLAIR_SECRET not in result["agent_contents"]["agent-001"]
    assert ALICE_SECRET not in result["agent_contents"]["agent-002"]
    for owner in ("agent-001", "agent-002"):
        audit = owner_audit(root, owner)
        assert audit["normalization"]["status"] == "completed"
        assert audit["normalization"]["parent_digest"] == audit["parent_proposal"]["digest"]
        assert audit["normalization"]["child_digest"] == audit["proposal"]["digest"]
        assert audit["normalization"]["lineage"][0]["parent_claim_id"] == "c1"
        assert audit["review"]["proposal_digest"] == audit["proposal"]["digest"]
        assert audit["retrieval"]["strategy"] == "all" and not audit["retrieval"]["truncated"]
        ws = Workspace.open(root / "owners" / owner / "workspace")
        task = Task.load(ws, audit["task_id"])
        assert task.data["status"] == "done" and task.log_path(1).is_file()
        assert read_json(ws.abspath(f"generated/responses/{audit['request_id']}.json")) == audit["response"]
    executor_ws = Workspace.open(root / "owners/agent-001/workspace")
    assert Task.load(executor_ws, result["decision_task_id"]).data["status"] == "done"
    assert (executor_ws.root / "generated/collaboration/answer.txt").read_text().strip() == result["final_response"]
    with pytest.raises(RunFailed, match="already has a run"):
        runtime.run(root, model)
    assert len(model.calls) == 7


# Retain the incumbent service, exact original prompts, and baseline version label.
def test_governed_comparator_does_not_normalize_or_change_extraction(tmp_path):
    root = prepared(tmp_path, treatment="governed-v1")
    model = IntegratedModel()
    result = runtime.run(root, model, treatment="governed-v1")
    assert result["status"] == "completed" and result["config"]["system_version"] == "governed-v1"
    assert [call["system"] for call in model.calls] == [EXTRACT, REVIEW, EXTRACT, REVIEW, DECIDE]
    assert result["usage"]["calls"] == 5
    assert "normalization" not in owner_audit(root)


# Preserve malformed normalization and stop before review, delivery, or any inference retry.
def test_invalid_normalization_is_a_persisted_failure(tmp_path):
    root = prepared(tmp_path)
    model = IntegratedModel(invalid_normalization=True)
    with pytest.raises(RunFailed):
        runtime.run(root, model)
    result = read_json(root / "run.json")
    audit = owner_audit(root)
    assert result["status"] == "failed" and result["responses"] == []
    assert [call["phase"] for call in model.calls] == ["extract", "normalize_release"]
    assert audit["normalization"]["status"] == "failed"
    assert audit["normalization"]["response"]["revisions"][0]["parent_claim_id"] == "unknown-parent"
    assert audit["events"][1]["response"] == audit["normalization"]["response"]
    assert audit["status"] == "failed" and audit["usage"]["calls"] == 2
    assert result["usage"]["calls"] == 2
    assert not (root / "owners/agent-001/workspace/generated/responses").exists()
    with pytest.raises(RunFailed, match="already has a run"):
        runtime.run(root, model)
    assert len(model.calls) == 2


# Recheck source versions at release even after normalization and model approval succeed.
def test_current_source_gate_survives_integration(tmp_path):
    root = prepared(tmp_path)
    source = root / "owners/agent-001/workspace/inputs/artifact-00001.txt"

    # Change the owner source during its review without altering the frozen proposal.
    def mutate(payload):
        source.write_text("A changed source after normalization.", encoding="utf-8")

    model = IntegratedModel(before_review=mutate)
    with pytest.raises(RunFailed):
        runtime.run(root, model)
    result = read_json(root / "run.json")
    audit = owner_audit(root)
    assert result["status"] == "failed" and result["responses"] == []
    assert audit["normalization"]["status"] == "completed" and audit["status"] == "failed"
    assert len(model.calls) == 3


# Treat a normalizer's complete omission as an empty release, without inventing a reviewer call.
def test_empty_normalized_proposal_remains_an_honest_completed_run(tmp_path):
    root = prepared(tmp_path)
    model = IntegratedModel(omit_all=True)
    result = runtime.run(root, model)
    assert result["status"] == "completed" and result["insufficient_evidence"] is True
    assert result["gathered_info"] == ""
    assert [call["phase"] for call in model.calls] == ["extract", "normalize_release"] * 2 + ["decide"]
    assert owner_audit(root)["review"]["decisions"] == []


# Reject full-corpus size violations before inference rather than dropping evidence units.
@pytest.mark.parametrize("bound", ["units", "bytes"])
def test_owner_bounds_fail_without_truncation_or_calls(tmp_path, monkeypatch, bound):
    case = projection()
    case["artifacts"]["agent-001"].append({"artifact_id": "artifact-extra", "type": "note",
                                           "text": "Owner-local extra fact.", "author": "Alice", "timestamp": ""})
    root = prepared(tmp_path, case=case)
    monkeypatch.setattr(runtime, "MAX_OWNER_EVIDENCE_UNITS" if bound == "units" else "MAX_INPUT_BYTES", 1)
    model = IntegratedModel()
    with pytest.raises(RunFailed):
        runtime.run(root, model)
    audit = owner_audit(root)
    assert audit["status"] == "failed" and model.calls == []
    assert len(audit["retrieved"]["units"]) == 2
    assert audit["retrieval"]["truncated"] is False


# Require preparation to identify the selected treatment before opening services.
def test_mismatched_preparation_starts_no_model_call(tmp_path):
    root = prepared(tmp_path, treatment="governed-v1")
    model = IntegratedModel()
    with pytest.raises(RunFailed, match="settings"):
        runtime.run(root, model, treatment="integrated")
    assert not (root / "run.json").exists() and model.calls == []


# Stop between model phases without changing an in-flight provider call or replaying it.
def test_call_gate_cancels_future_inference_and_preserves_usage():
    model = IntegratedModel()
    cancelled = threading.Event()
    gate = runtime._CallGate(model, cancelled)
    gate.complete_json(json.dumps({"gathered_info": "", "evidence_refs": []}), system=DECIDE)
    assert gate.usage is model.usage
    cancelled.set()
    with pytest.raises(RunFailed, match="no additional inference"):
        gate.complete_json(json.dumps({"gathered_info": "", "evidence_refs": []}), system=DECIDE)
    assert model.usage["calls"] == 1


# Apply a longer explicit deadline to real owner exchanges without changing inference settings.
def test_owner_http_deadline_is_explicit_and_native_tasks_wait_for_stop(tmp_path, monkeypatch):
    root = prepared(tmp_path)
    original = runtime.request_json
    deadlines = []

    # Observe the real call arguments while still performing the authenticated HTTP request.
    def observe(url, token, data=None, **kwargs):
        if url.endswith("/request"):
            deadlines.append(kwargs.get("timeout"))
        return original(url, token, data, **kwargs)

    monkeypatch.setattr(runtime, "request_json", observe)
    result = runtime.run(root, IntegratedModel())
    assert deadlines == [runtime.OWNER_REQUEST_TIMEOUT_SECONDS] * 2
    assert runtime.OWNER_REQUEST_TIMEOUT_SECONDS > 3 * 120
    assert result["decision_task_id"]


# Settle an in-flight call after transport failure and block the remaining owner phases.
def test_timeout_settles_private_receipt_before_returning(tmp_path, monkeypatch):
    root = prepared(tmp_path)
    entered, release_call = threading.Event(), threading.Event()
    response_threads = []
    original_request = runtime.request_json
    original_stop = runtime.ArtifactService.stop

    # Hold one synthetic provider call until the coordinator begins its shutdown path.
    class BlockingModel(IntegratedModel):
        # Finish exactly the already-started extraction after service shutdown starts.
        def complete_json(self, prompt, system=None):
            if system.startswith("PHASE: extract"):
                entered.set()
                assert release_call.wait(5)
            return super().complete_json(prompt, system=system)

    # Keep the genuine HTTP request active while simulating a lost coordinator response.
    def request_with_timeout(url, token, data=None, **kwargs):
        if not url.endswith("/request"):
            return original_request(url, token, data, **kwargs)

        # Consume the eventual terminal HTTP failure without converting it into a peer response.
        def background_request():
            try:
                original_request(url, token, data, **kwargs)
            except RunFailed:
                pass

        thread = threading.Thread(target=background_request)
        response_threads.append(thread)
        thread.start()
        assert entered.wait(5)
        raise TimeoutError("Synthetic lost response")

    # Allow the already-started call to settle only after cancellation has been signaled.
    def stop_and_release(service):
        release_call.set()
        return original_stop(service)

    monkeypatch.setattr(runtime, "request_json", request_with_timeout)
    monkeypatch.setattr(runtime.ArtifactService, "stop", stop_and_release)
    model = BlockingModel()
    with pytest.raises(RunFailed):
        runtime.run(root, model)
    for thread in response_threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    saved = read_json(root / "run.json")
    audit = owner_audit(root)
    assert saved["status"] == audit["status"] == "failed"
    assert saved["responses"] == [] and saved["usage"]["calls"] == 1
    assert [call["phase"] for call in model.calls] == ["extract"]
    assert audit["normalization"]["status"] == "failed"
    assert audit["events"][-1]["status"] == "failed"
    assert "settlement_error" not in saved
