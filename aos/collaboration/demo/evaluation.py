# Display saved official scores without passing evaluator data to an actor.

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from .view import build_native_view


# Read a bounded operator-selected JSON object and retain its byte digest.
def _read(path):
    raw = Path(path).expanduser().resolve().read_bytes()
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("Evaluation attachment is too large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Evaluation attachment must be an object")
    return value, hashlib.sha256(raw).hexdigest()


# Preserve finite numeric scores while rejecting booleans and misleading string values.
def _number(value):
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return None


# Match the supplied export to the recorded requester-visible answer and communication trace.
def _match_pipeline(pipeline, view, metadata):
    config = pipeline.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Invalid pipeline configuration")
    if config.get("case_id") != view["scenario"]["case_id"]:
        raise ValueError("Pipeline case does not match the replay")
    if pipeline.get("final_response") != view["answer"]["text"]:
        raise ValueError("Pipeline answer does not match the replay")
    if config.get("run_index") != metadata.get("run_index"):
        raise ValueError("Pipeline run index does not match the replay")
    if config.get("agent_llm") != metadata.get("model"):
        raise ValueError("Pipeline actor model does not match the replay")
    if config.get("system") != "aos_artifact_requests":
        raise ValueError("This replay importer attaches AOS artifact-request evaluations only")
    if pipeline.get("task_description") != view["scenario"]["task"]:
        raise ValueError("Pipeline task does not match the replay")
    expected_gathered = "\n".join(
        f"{claim['text']} [evidence: {', '.join(claim['evidence_refs'])}]" for claim in view["evidence"])
    if pipeline.get("gathered_info") != expected_gathered:
        raise ValueError("Pipeline gathered evidence does not match the replay")
    people = {person["id"]: person["name"] for person in view["participants"]}
    logs = pipeline.get("log")
    if not isinstance(logs, list) or len(logs) != len(view["timeline"]):
        raise ValueError("Pipeline message count does not match the replay")
    for saved, event in zip(logs, view["timeline"]):
        if not isinstance(saved, dict):
            raise ValueError("Invalid pipeline log record")
        if saved.get("sender") != people[event["sender_id"]] or saved.get("receiver") != people[event["recipient_id"]]:
            raise ValueError("Pipeline sender or recipient does not match the replay")
        if saved.get("request_id") != event["request_id"] or saved.get("kind") != event["kind"]:
            raise ValueError("Pipeline message identity does not match the replay")
        if event["kind"] == "request":
            if saved.get("message") != event["body"]:
                raise ValueError("Pipeline request does not match the replay")
        elif event.get("response") is None or json.loads(saved.get("message", "null")) != event["response"]:
            raise ValueError("Pipeline response does not match the replay")


# Extract only aggregate metrics and attribute IDs, never gold values, private prompts, or raw judgments.
def attach_evaluation(evaluation_path, pipeline_path, view, metadata):
    if pipeline_path is None:
        raise ValueError("Attach the matching pipeline export with the saved judgment")
    result, judgment_hash = _read(evaluation_path)
    pipeline, pipeline_hash = _read(pipeline_path)
    _match_pipeline(pipeline, view, metadata)
    config = result.get("config", {})
    source = pipeline["config"]
    for field in ("scenario", "system", "agent_llm"):
        if config.get(field) != source.get(field):
            raise ValueError("Judgment metadata does not match the selected export")
    return _evaluation_display(result, pipeline_hash, judgment_hash)


# Bind a native judgment to its exact selected pipeline without inventing an AOS actor record.
def attach_native_evaluation(evaluation_path, pipeline_path, view, metadata, *, case):
    result, judgment_hash = _read(evaluation_path)
    pipeline, pipeline_hash = _read(pipeline_path)
    if pipeline_hash != view["source_sha256"]:
        raise ValueError("Native evaluation pipeline differs from the selected replay")
    candidate = build_native_view(pipeline, view["run_id"], pipeline_hash, case=case)
    if candidate != view:
        raise ValueError("Native evaluation pipeline does not match the replay")
    config, source = result.get("config", {}), pipeline["config"]
    if not isinstance(config, dict):
        raise ValueError("Invalid native judgment metadata")
    for field in ("scenario", "system", "agent_llm"):
        if config.get(field) != source.get(field):
            raise ValueError("Native judgment metadata does not match the selected export")
    if config.get("run_index", metadata["run_index"]) != metadata["run_index"]:
        raise ValueError("Native judgment run index differs from the selected replay")
    if source["agent_llm"] != metadata["model"]:
        raise ValueError("Native actor model differs from the selected replay")
    return _evaluation_display(result, pipeline_hash, judgment_hash, native=True)


# Minimize saved evaluator output identically for AOS and native replay records.
def _evaluation_display(result, pipeline_hash, judgment_hash, *, native=False):
    config = result.get("config", {})
    checks = result.get("verification", {})
    if not isinstance(checks, dict):
        checks = {}
    # An old evaluator file usually has no export digest (operator attachment remains an explicit assertion).
    rows = []
    metrics = {}
    utility = _number(result.get("utility", {}).get("score"))
    if utility is not None:
        metrics["utility"] = {"score": utility}
        rows.append({"label": "Decision utility", "value": str(utility)})
    comp = result.get("completeness", {})
    numerator, denominator = _number(comp.get("n_covered")), _number(comp.get("n_approp"))
    rate = _number(comp.get("rate"))
    metrics["completeness"] = {"covered": numerator, "eligible": denominator, "rate": rate}
    rows.append({"label": "Required facts gathered", "value": f"{numerator} / {denominator}"})
    for surface, label in (("V_G", "Gathering flags"), ("V_A2A", "Peer-message flags"),
                           ("V_out", "Final-answer flags"), ("V_A", "Agent-context flags")):
        item = result.get(surface)
        if not isinstance(item, dict):
            continue
        # Agent-context scoring stores the union across agents under total (with its own denominator).
        if surface == "V_A" and isinstance(item.get("total"), dict):
            item = item["total"]
        flagged = [x for x in item.get("flagged", []) if isinstance(x, str) and len(x) <= 80]
        nflagged = _number(item.get("n_flagged"))
        eligible = _number(item.get("n_inapp", item.get("n_hidden", item.get("n_potential"))))
        metrics[surface] = {"flagged": flagged, "n_flagged": nflagged, "eligible": eligible,
                            "rate": _number(item.get("rate"))}
        display = f"{nflagged} / {eligible}" if nflagged is not None and eligible is not None else "Unavailable in saved judgment"
        rows.append({"label": label, "value": display})
    return {"status": "attached", "source": "saved_native_official_judgment" if native else "saved_official_judgment", "binding": "operator_attached",
            "pipeline_content_checked": True, "pipeline_sha256": pipeline_hash,
            "judgment_sha256": judgment_hash,
            "binding_note": ("The exact native pipeline is the source of this recorded answer and transcript. " if native else
                             "The export matches this recorded answer and message trace. ") +
                            "The historical judgment has no cryptographic export binding; its association is asserted by the importing operator.",
            "display_note": ("Saved native PiSAs evaluation; no AOS owner-release audit is implied. " if native else "") +
                            "Official evaluator findings, not live privacy guarantees. Attribute IDs are shown; gold values and private audit text are omitted.",
            "judge": config.get("judge_llm"), "judge_mode": config.get("judge_mode"),
            "agent_audit": config.get("agent_audit") is True,
            "output_audit": config.get("output_audit") is True,
            "verification": {k: checks[k] for k in ("checks", "abstained", "votes_missing", "degraded")
                             if type(checks.get(k)) in (int, bool)},
            "rows": rows, "metrics": metrics}
