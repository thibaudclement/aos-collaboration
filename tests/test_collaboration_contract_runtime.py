# Exercise the prospective coordinator with real owner services and exact model-audit preservation.

from copy import deepcopy
import json

import pytest

from aos.collaboration.common import read_json, write_json
from aos.collaboration.integrated import contract, contract_runtime as runtime
from aos.collaboration.integrated.components import extraction_system
from aos.collaboration.model import DECIDE, EXTRACT, REVIEW
from aos.collaboration.integrated.components import NORMALIZE
from aos.collaboration.runtime import RunFailed
from test_collaboration_integrated_runtime import IntegratedModel, owner_audit, prepared


# Return a controlled task sidecar without altering the actual system or payload received.
class ContractModel(IntegratedModel):
    # Select valid task structure or a concrete contract failure in the one decision response.
    def __init__(self, *, bad=None, **kwargs):
        super().__init__(**kwargs)
        self.bad = bad
        self.raw_decision = None

    # Account for the inherited call once and retain its exact raw response independently.
    def complete_json(self, prompt, system=None):
        raw = super().complete_json(prompt, system=system)
        if system == DECIDE:
            raw["schedule"] = {"FIXTURE-MEETING": {"slot": "Tuesday at 10:00", "room": "fixture-room"}}
            if self.bad == "refs":
                raw["used_evidence_refs"] = ["ref-not-released"]
            elif self.bad == "private_sidecar":
                raw["debug"] = "PRIVATE_AUXILIARY_SENTINEL"
            self.raw_decision = deepcopy(raw)
        return raw


# Declare this prospective profile in the prepared manifest before any execution begins.
def prepare_contract(tmp_path, treatment="integrated"):
    root = prepared(tmp_path, treatment=treatment)
    manifest = read_json(root / "manifest.json")
    manifest["config"]["response_contract"] = contract.specification()
    write_json(root / "manifest.json", manifest)
    return root


# Use one common adapter while preserving each arm's exact prompts, owner path and call count.
@pytest.mark.parametrize("treatment,count", [("integrated", 7), ("governed-v1", 5)])
def test_both_arms_preserve_actual_model_io(tmp_path, treatment, count):
    root = prepare_contract(tmp_path, treatment)
    model = ContractModel()
    result = runtime.run(root, model, treatment=treatment)
    assert result == read_json(root / "run.json") and result["status"] == "completed"
    assert result["usage"]["calls"] == len(model.calls) == count
    assert result["config"]["response_contract"] == contract.specification()
    expected = ({"extract": extraction_system("cap24"), "normalize_release": NORMALIZE,
                 "review": REVIEW, "decide": DECIDE} if treatment == "integrated" else
                {"extract": EXTRACT, "review": REVIEW, "decide": DECIDE})
    assert all(call["system"] == expected[call["phase"]] for call in model.calls)
    event = result["decision_audit"][0]
    assert event["response"] == model.raw_decision
    assert event["prompt"] == model.calls[-1]["prompt"]
    assert event["system"] == model.calls[-1]["system"] == DECIDE
    assert set(json.loads(event["prompt"])) == {"task", "recipient", "participants", "gathered_info", "evidence_refs"}
    assert json.dumps(model.raw_decision, ensure_ascii=False) in result["agent_contents"]["agent-001"]
    answer = json.loads(result["final_response"])
    assert answer == {key: value for key, value in model.raw_decision.items()
                      if key not in {"used_evidence_refs", "insufficient_evidence"}}
    assert result["decision_contract"]["operation"] == "serialize_answer_fields"
    assert result["decision_contract"]["raw_response_sha256"] == contract.document_sha256(model.raw_decision)
    assert result["n_a2a"] == 2 and result["decision_task_id"]
    if treatment == "integrated":
        assert owner_audit(root)["normalization"]["status"] == "completed"
    else:
        assert "normalization" not in owner_audit(root)
    with pytest.raises(RunFailed, match="already has a run"):
        runtime.run(root, model, treatment=treatment)
    assert len(model.calls) == count


# Retain an unusable paid-response analogue as failure with raw response and charge accounting intact.
@pytest.mark.parametrize("bad", ["refs", "private_sidecar"])
def test_contract_failure_preserves_raw_response_and_forbids_retry(tmp_path, bad):
    root = prepare_contract(tmp_path)
    model = ContractModel(bad=bad)
    with pytest.raises(RunFailed):
        runtime.run(root, model)
    result = read_json(root / "run.json")
    assert result["status"] == "failed" and result["final_response"] == ""
    assert result["decision_audit"][0]["response"] == model.raw_decision
    assert result["decision_audit"][0]["status"] == "ok"
    assert result["decision_contract"]["status"] == "rejected"
    assert result["usage"]["calls"] == 7 and result["decision_usage"]["calls"] == 1
    assert len(result["responses"]) == 2 and result["n_a2a"] == 2
    assert "decision_task_id" not in result
    with pytest.raises(RunFailed, match="already has a run"):
        runtime.run(root, model)
    assert len(model.calls) == 7


# Refuse accidental use of a new contract under an undeclared preparation.
def test_profile_must_be_declared_before_any_service_or_call(tmp_path):
    root = prepared(tmp_path)
    model = ContractModel()
    with pytest.raises(RunFailed, match="response contract"):
        runtime.run(root, model)
    assert model.calls == [] and not (root / "run.json").exists()


# Preserve owner failure semantics rather than reaching the answer adapter after malformed normalization.
def test_owner_failure_never_reaches_adapter(tmp_path):
    root = prepare_contract(tmp_path)
    model = ContractModel(invalid_normalization=True)
    with pytest.raises(RunFailed):
        runtime.run(root, model)
    result = read_json(root / "run.json")
    assert result["decision_contract"]["status"] == "not_started"
    assert result["decision_audit"] == [] and model.raw_decision is None
    assert result["usage"]["calls"] == 2
