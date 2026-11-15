# Check lossless prospective serialization without changing task choices or original validation.

from copy import deepcopy
import json

import pytest

from aos.collaboration.integrated import contract
from aos.collaboration.runtime import RunFailed, _decision


# Supply valid common controls independently of any benchmark answer or hidden label.
def envelope(final="An authored fixture answer.", **extras):
    return {"final_response": final, "used_evidence_refs": ["ref-fixture"],
            "insufficient_evidence": False, **extras}


# Preserve ordinary historical output byte-for-byte instead of gratuitously reformatting it.
def test_string_identity_keeps_whitespace_unicode_and_controls():
    raw = envelope('  Héllo "team".\nUse C:\\tmp.  ')
    before = deepcopy(raw)
    accepted, audit = contract.adapt(raw, {"ref-fixture"})
    assert accepted == before == raw and accepted is not raw
    assert audit["operation"] == "identity" and audit["provider_requests"] == 0
    assert audit["raw_response_sha256"] == audit["accepted_envelope_sha256"]


# Reproduce the observed schedule sibling failure and preserve it prospectively without rewriting raw data.
def test_schedule_sibling_roundtrips_without_changing_original_validator():
    raw = envelope("Use the supplied window.", schedule={"MEET-001": {
        "slot": "Tuesday 10:00-11:00", "room": "cedar"}})
    before = deepcopy(raw)
    with pytest.raises(RunFailed, match="schema"):
        _decision(raw, {"ref-fixture"})
    accepted, audit = contract.adapt(raw, {"ref-fixture"})
    assert _decision(accepted, {"ref-fixture"}) == accepted
    assert json.loads(accepted["final_response"]) == {
        "final_response": before["final_response"], "schedule": before["schedule"]}
    assert raw == before and "TUE_1000" not in accepted["final_response"]
    assert audit["task_sidecar_fields"] == ["schedule"]


# Support structured answer values while preserving nested strings, arrays, nulls and booleans.
@pytest.mark.parametrize("final", [
    {"assignment": {"TASK-TEST": ["A", "B"]}, "explanation": 'Quote "x"\nµ',
     "metadata": {"unresolved": None, "conditions": [True, False]}},
    [{"decision": "authored example", "reason": "line\nbreak"}],
])
def test_structured_final_roundtrips_without_semantic_validation(final):
    raw = envelope(final)
    accepted, audit = contract.adapt(raw, {"ref-fixture"})
    assert json.loads(accepted["final_response"]) == final
    assert audit["operation"] == "serialize_structured_final"
    assert accepted["used_evidence_refs"] == raw["used_evidence_refs"]


# Preserve contradictory task claims rather than selecting the apparent better answer.
def test_contradictory_text_and_structure_are_both_retained():
    raw = envelope("Choose A.", assignment={"TASK": ["B"]}, explanation="Fixture contradiction.")
    accepted, _ = contract.adapt(raw, {"ref-fixture"})
    answer = json.loads(accepted["final_response"])
    assert answer == {key: raw[key] for key in ("final_response", "assignment", "explanation")}


# Treat insufficient evidence as an honest answer state rather than an execution failure.
def test_insufficient_evidence_and_empty_reference_list_remain_unchanged():
    raw = envelope({"decision": "Not established", "missing": ["authored prerequisite"]})
    raw.update(used_evidence_refs=[], insufficient_evidence=True)
    accepted, audit = contract.adapt(raw, set())
    assert accepted["insufficient_evidence"] is True and accepted["used_evidence_refs"] == []
    assert audit["uncertainty_unchanged"] and audit["references_unchanged"]


# Never infer missing required fields or accept unknown references and mistyped controls.
@pytest.mark.parametrize("mutation", [
    lambda row: row.pop("used_evidence_refs"),
    lambda row: row.pop("insufficient_evidence"),
    lambda row: row.pop("final_response"),
    lambda row: row.update(used_evidence_refs=["ref-unknown"]),
    lambda row: row.update(used_evidence_refs="ref-fixture"),
    lambda row: row.update(used_evidence_refs=[False]),
    lambda row: row.update(insufficient_evidence="false"),
    lambda row: row.update(insufficient_evidence=0),
])
def test_invalid_controls_still_fail(mutation):
    raw = envelope(schedule={"fixture": "value"})
    mutation(raw)
    before = deepcopy(raw)
    with pytest.raises(contract.ContractError):
        contract.adapt(raw, {"ref-fixture"})
    assert raw == before


# Unsupported auxiliary fields must not be silently discarded or silently published.
@pytest.mark.parametrize("field", ["analysis", "debug", "private_reason", "used_refs", "unknown"])
def test_unsupported_sidecars_are_explicit_failures(field):
    raw = envelope(**{field: "PRIVATE_AUXILIARY_SENTINEL"})
    with pytest.raises(contract.ContractError, match="unsupported"):
        contract.adapt(raw, {"ref-fixture"})
    assert raw[field] == "PRIVATE_AUXILIARY_SENTINEL"


# Keep the existing final text bound and refuse lossy or empty structured representations.
@pytest.mark.parametrize("final", ["", "  ", {}, [], None, False, 4, "x" * 16001,
                                    {"value": "x" * 16000}, {1: "integer key"},
                                    {"value": float("nan")}, {"value": (1, 2)}])
def test_empty_oversize_or_nonjson_answers_fail(final):
    with pytest.raises(contract.ContractError):
        contract.adapt(envelope(final), {"ref-fixture"})


# Validate the encoded representation length rather than truncating content to fit the limit.
def test_sidecar_encoding_preserves_limit_as_a_failure():
    raw = envelope("x" * 15999, explanation="another field")
    with pytest.raises(contract.ContractError):
        contract.adapt(raw, {"ref-fixture"})
    assert len(raw["final_response"]) == 15999


# Preserve the boundary value admitted by the inherited text validator.
def test_existing_text_limit_is_still_admitted_exactly():
    raw = envelope("x" * 16000)
    accepted, _ = contract.adapt(raw, {"ref-fixture"})
    assert accepted == raw


# Make the prospective method explicit without altering model instructions or adding calls.
def test_specification_has_no_prompt_or_inference_change():
    spec = contract.specification()
    assert spec["id"] == "aos-response-contract-v1"
    assert spec["original_decide_prompt_changed"] is spec["model_request_changed"] is False
    assert spec["additional_model_calls"] == 0 and spec["inference_retry"] is False
