# Preserve supported task-answer structure inside the existing three-field AOS envelope.

from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from ..runtime import RunFailed, _decision

CONTRACT_ID = "aos-response-contract-v1"
CONTROL_FIELDS = frozenset({"final_response", "used_evidence_refs", "insufficient_evidence"})
TASK_FIELDS = frozenset({"assignment", "schedule", "explanation", "decision", "outcome",
                         "disposition", "severity", "severity_tier", "justification", "readiness"})


# Reject unsupported representation changes without returning a repaired semantic answer.
class ContractError(RunFailed):
    pass


# Identify the deterministic representation rule separately from each component treatment.
def specification() -> dict:
    return {
        "id": CONTRACT_ID,
        "original_decide_prompt_changed": False,
        "model_request_changed": False,
        "mandatory_fields": sorted(CONTROL_FIELDS),
        "permitted_task_sidecars": sorted(TASK_FIELDS),
        "string_without_sidecars": "Preserve the original final_response string exactly.",
        "structured_final_without_sidecars": "Serialize a nonempty JSON object or array as final_response text.",
        "task_sidecars": "Serialize original final_response and every permitted sibling answer field together as final_response text.",
        "unsupported_fields": "Reject and retain the raw failed attempt; never silently drop a field.",
        "references_and_uncertainty": "Retain unchanged and validate with the original AOS decision validator.",
        "semantic_policy": "Do not add identifiers, choose between contradictions, infer facts, or change decisions.",
        "inference_retry": False,
        "additional_model_calls": 0,
        "original_final_text_character_limit": 16000,
    }


# Reject non-JSON values and lossy key coercion before performing a representation change.
def _json_value(value, *, depth=0) -> None:
    if depth > 100:
        raise ContractError("Decision answer exceeds the JSON nesting bound")
    if value is None or type(value) in {bool, int}:
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ContractError("Decision answer contains invalid Unicode") from None
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _json_value(key, depth=depth + 1)
            _json_value(item, depth=depth + 1)
        return
    raise ContractError("Decision answer is not losslessly representable as JSON")


# Hash the preserved JSON value without attributing a deterministic encoding to the model.
def document_sha256(value) -> str:
    _json_value(value)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# Encode supported task content while preserving every value and the original field order.
def _answer_text(value) -> str:
    _json_value(value)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise ContractError("Decision answer cannot be serialized") from None


# Apply only the declared lossless representation change, then reuse the original validator.
def adapt(response: dict, refs: set[str]) -> tuple[dict, dict]:
    if not isinstance(response, dict) or not all(isinstance(key, str) for key in response):
        raise ContractError("Decision response must be a JSON object with string keys")
    if not CONTROL_FIELDS <= set(response):
        raise ContractError("Decision response is missing a mandatory AOS control field")
    sidecars = set(response) - CONTROL_FIELDS
    if not sidecars <= TASK_FIELDS:
        raise ContractError("Decision response contains an unsupported top-level field")
    _json_value(response)
    final = response["final_response"]
    if isinstance(final, str):
        if not final.strip():
            raise ContractError("Decision final_response must not be empty")
    elif not isinstance(final, (dict, list)) or not final:
        raise ContractError("Decision final_response must be nonempty text, object, or array")
    candidate = deepcopy({key: response[key] for key in
                          ("final_response", "used_evidence_refs", "insufficient_evidence")})
    if sidecars:
        task_answer = {key: value for key, value in response.items()
                       if key not in {"used_evidence_refs", "insufficient_evidence"}}
        candidate["final_response"] = _answer_text(task_answer)
        operation = "serialize_answer_fields"
    elif isinstance(final, (dict, list)):
        candidate["final_response"] = _answer_text(final)
        operation = "serialize_structured_final"
    else:
        operation = "identity"
    try:
        accepted = _decision(candidate, refs)
    except RunFailed as exc:
        raise ContractError(str(exc)) from None
    audit = {"id": CONTRACT_ID, "status": "accepted", "operation": operation,
             "raw_response_sha256": document_sha256(response),
             "accepted_envelope_sha256": document_sha256(accepted),
             "retained_answer_fields": [key for key in response
                                        if key not in {"used_evidence_refs", "insufficient_evidence"}],
             "task_sidecar_fields": sorted(sidecars),
             "input_final_response_type": type(final).__name__,
             "references_unchanged": accepted["used_evidence_refs"] == response["used_evidence_refs"],
             "uncertainty_unchanged": accepted["insufficient_evidence"] is response["insufficient_evidence"],
             "content_policy": "Representation only; no added facts, selected alternative, or inferred identifier.",
             "provider_requests": 0}
    return accepted, audit
