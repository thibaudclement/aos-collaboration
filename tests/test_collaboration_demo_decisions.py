# Verify optional observed decisions without treating synthetic fixtures as benchmark performance.

from __future__ import annotations
import copy
import json
import pytest
from aos.collaboration.demo.decisions import (
    OBSERVED_DECIDE, OBSERVED_PROTOCOL, call_observed, released_catalogue, validate_observed,
)
from aos.collaboration.demo.runner import execute_case
from aos.collaboration.model import ModelFailure, visible_context
from aos.collaboration.runtime import RunFailed

REF_A = "ref-" + "a" * 24
REF_B = "ref-" + "b" * 24
PRIVATE = "PRIVATE_OWNER_DETAIL_NEVER_RELEASED"


# Construct two actual-release-shaped records that disagree about the same room capacity.
def release_records():
    result = []
    for owner, request, text, ref in (
        ("agent-001", "req-one", "Room A seats 20 people in the current layout.", REF_A),
        ("agent-002", "req-two", "Room A seats 30 people in the current layout.", REF_B),
    ):
        response = {"request_id": request, "status": "released", "answer": text,
                    "claims": [{"text": text, "evidence_refs": [ref]}], "coverage": "not_established"}
        result.append({"owner_id": owner, "response": response})
    return result


# Define an unresolved-but-answered response to ensure disagreement alone never crashes execution.
def unresolved_response():
    return {"final_response": "Both reports support seating at least 20 people; the exact capacity remains disputed.",
            "used_evidence_refs": [REF_A, REF_B], "insufficient_evidence": False,
            "decision_action": "unresolved_but_proceed",
            "decision_note": "The exact capacity differs, but both claims satisfy the requested minimum of 20.",
            "disagreements": [{"kind": "factual_conflict",
                "claim_ids": ["agent-001:req-one:c1", "agent-002:req-two:c1"],
                "scope": "Room A, current layout, seating capacity", "material": False,
                "action": "unresolved_but_proceed", "selected_claim_ids": [],
                "reason": "The sources give 20 and 30 seats; neither source supersedes the other."}]}


# Return a fixed response while capturing the exact decision call for independent boundary assertions.
class RecordingDecision:
    # Retain the predetermined output and each input without invoking an API.
    def __init__(self, output):
        self.output = output
        self.calls = []

    # Record the requested payload and return an independent response object.
    def complete_json(self, prompt, system=None):
        self.calls.append({"prompt": prompt, "system": system})
        return copy.deepcopy(self.output)


# A remaining conflict may be explicitly recorded while the model still returns a usable answer.
def test_unresolved_conflict_does_not_automatically_block_answer():
    catalogue = released_catalogue(release_records())
    decision, observation = validate_observed(unresolved_response(), catalogue)
    assert "at least 20" in decision["final_response"]
    assert not decision["insufficient_evidence"]
    assert observation["decision_action"] == "unresolved_but_proceed"
    assert observation["disagreements"][0]["selected_claim_ids"] == []
    assert observation["disagreements"][0]["claim_ids"] == [
        "agent-001:req-one:c1", "agent-002:req-two:c1"]
    assert observation["disagreements"][0]["evidence_refs"] == [REF_A, REF_B]


# A recorded resolution keeps the rejected claim and exposes the evidence behind the model's choice.
def test_resolved_conflict_keeps_both_claims_and_the_model_selection():
    value = unresolved_response()
    value.update(decision_action="resolved", final_response="The room seats 20 people.")
    entry = value["disagreements"][0]
    entry.update(action="resolved", material=True, selected_claim_ids=["agent-001:req-one:c1"],
                 reason="This is a synthetic assessment for testing reference retention, not verified truth.")
    decision, observation = validate_observed(value, released_catalogue(release_records()))
    assert decision["final_response"] == value["final_response"]
    assert len(observation["disagreements"][0]["claim_ids"]) == 2
    assert observation["disagreements"][0]["selected_claim_ids"] == ["agent-001:req-one:c1"]


# Clearly compatible evidence can yield an ordinary answer without inventing a disagreement.
def test_no_disagreement_keeps_the_public_task_output_separate_from_the_note():
    value = {"final_response": "ready", "used_evidence_refs": [REF_A], "insufficient_evidence": False,
             "decision_action": "no_disagreement", "decision_note": "The supplied condition is satisfied.",
             "disagreements": []}
    decision, observation = validate_observed(value, released_catalogue(release_records()))
    assert decision["final_response"] == "ready"
    assert "condition" not in decision["final_response"]
    assert observation["disagreements"] == []


# An honest evidence-gap response is a semantic outcome, not a malformed output or provider failure.
def test_insufficient_evidence_remains_a_successfully_validated_answer():
    value = unresolved_response()
    value.update(decision_action="insufficient_evidence", insufficient_evidence=True,
                 final_response="The evidence does not establish whether the room seats 25.")
    value["disagreements"][0].update(action="insufficient_evidence", material=True)
    decision, observation = validate_observed(value, released_catalogue(release_records()))
    assert decision["insufficient_evidence"]
    assert observation["decision_action"] == "insufficient_evidence"
    assert decision["final_response"]


# Unknown evidence, fabricated winners, inconsistent flags and extra fields cannot enter the accepted record.
@pytest.mark.parametrize("mutation", [
    lambda r: r.update(used_evidence_refs=["ref-" + "c" * 24]),
    lambda r: r["disagreements"][0].update(claim_ids=["invented", "agent-001:req-one:c1"]),
    lambda r: r["disagreements"][0].update(selected_claim_ids=["agent-001:req-one:c1"]),
    lambda r: r["disagreements"][0].update(material="false"),
    lambda r: r.update(insufficient_evidence=True),
    lambda r: r.update(decision_action="resolved") or r["disagreements"][0].update(material=True),
    lambda r: r.update(private_reason=PRIVATE),
])
def test_invalid_observations_are_rejected_without_output_repair(mutation):
    value = unresolved_response()
    mutation(value)
    with pytest.raises(RunFailed):
        validate_observed(value, released_catalogue(release_records()))


# The model sees one actual combined call and the audit preserves it, instead of fabricating a legacy prompt.
def test_single_call_exact_audit_and_released_only_payload():
    records = release_records()
    original = copy.deepcopy(records)
    catalogue = released_catalogue(records)
    model = RecordingDecision(unresolved_response())
    audit = []
    payload = {"task": "Can Room A seat at least 20?", "recipient": "Requester",
               "gathered_info": "\n".join(item["text"] for item in catalogue),
               "evidence_refs": [REF_A, REF_B]}
    decision, observation = call_observed(model, payload, catalogue, audit)
    assert len(model.calls) == 1
    assert audit[0]["system"] == model.calls[0]["system"] == OBSERVED_DECIDE
    assert audit[0]["prompt"] == model.calls[0]["prompt"]
    assert audit[0]["response"] == unresolved_response()
    assert json.loads(audit[0]["prompt"])["released_claims"] == catalogue
    assert PRIVATE not in visible_context(audit)
    assert records == original
    assert decision["final_response"] and observation["status"] == "recorded"


# A bad model response remains in the private audit and never becomes an accepted decision.
def test_failed_schema_preserves_raw_evidence_without_a_second_call():
    value = unresolved_response()
    value["disagreements"][0]["claim_ids"][0] = "invented"
    model = RecordingDecision(value)
    audit = []
    with pytest.raises(ModelFailure):
        call_observed(model, {"task": "Test"}, released_catalogue(release_records()), audit)
    assert len(model.calls) == 1
    assert audit[0]["status"] == "failed"
    assert audit[0]["response"] == value


# Run genuine scripted owner subprocesses through both versions and verify privacy and version separation.
@pytest.mark.parametrize("version", ["governed-v1", "observed-v2"])
def test_scripted_live_flow_uses_real_services_without_a_provider(tmp_path, monkeypatch, version):
    # An accidental model-client construction must fail the test even if credentials are configured locally
    def forbid_client(*args, **kwargs):
        raise AssertionError("The offline fixture must not create an API model")
    monkeypatch.setattr("aos.collaboration.demo.runner.create_model", forbid_client)
    monkeypatch.setattr("aos.collaboration.runtime.create_model", forbid_client)
    root = tmp_path / version
    messages = []
    result = execute_case(None, root, version=version, progress=messages.append)
    saved = json.loads((root / "run.json").read_text())
    assert saved == result
    assert result["status"] == "completed"
    assert result["n_a2a"] == 2
    assert len(result["responses"]) == 2
    assert "10:00" in result["final_response"] and "10:30" in result["final_response"]
    requester_data = json.dumps({key: result.get(key) for key in (
        "events", "responses", "gathered_info", "final_response", "decision_audit", "decision_observation")})
    assert "DEMO_PRIVATE" not in requester_data
    assert messages and all(isinstance(message, str) for message in messages)
    if version == "observed-v2":
        assert result["config"]["protocol"] == OBSERVED_PROTOCOL
        assert result["decision_observation"]["disagreements"] == []
        assert result["decision_observation"]["decision_action"] == "no_disagreement"
        assert len(result["decision_audit"]) == 1
        assert result["decision_audit"][0]["system"] == OBSERVED_DECIDE
        assert result["decision_audit"][0]["prompt"] in result["agent_contents"]["agent-001"]
        assert len(result["released_claims"]) == 2
    else:
        assert result["config"]["protocol"] == "aos-artifact-requests-v1"
        assert "decision_observation" not in result
    # Existing immutable run directories cannot be re-executed under either version.
    with pytest.raises((ValueError, RunFailed)):
        execute_case(None, root, version=version)
