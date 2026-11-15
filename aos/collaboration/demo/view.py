# Project a saved artifact-request run into requester-visible replay data, without running an agent.

from __future__ import annotations
import hashlib
import json
import math
import re

RESPONSE_KEYS = {"request_id", "status", "answer", "claims", "coverage"}
EVENT_KEYS = {"sender_id", "recipient_id", "request_id", "kind", "body", "delivery", "timestamp"}
REF = re.compile(r"ref-[0-9a-f]{24}\Z")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
RESPONSE_CONTRACT_ID = "aos-response-contract-v1"


# Reject ambiguous imports rather than inventing a received message or an execution event.
class ReplayError(ValueError):
    pass


# Express structural conditions consistently for CLI and Python callers.
def require(condition, message):
    if not condition:
        raise ReplayError(message)


# Preserve a recorded string without trying to redact facts that actually reached the requester.
def string(value, label, *, nonempty=False):
    require(isinstance(value, str) and (bool(value.strip()) or not nonempty), "Invalid " + label)
    return value


# Identify the supported prospective serializer without exposing its private policy or execution audit.
def response_contract_id(config):
    if "response_contract" not in config:
        return None
    profile = config["response_contract"]
    require(isinstance(profile, dict) and profile.get("id") == RESPONSE_CONTRACT_ID,
            "Unsupported response contract profile")
    require(config.get("protocol") == "aos-artifact-requests-v1" and config.get("mode") == "governed"
            and config.get("system_version", "governed-v1") in {"governed-v1", "integrated-v3"},
            "Response contract does not match the recorded AOS treatment")
    return RESPONSE_CONTRACT_ID


# Make deterministic viewer-only identities (the original runtime never emitted these identifiers).
def view_id(run_id, source_sha256, *parts):
    body = json.dumps([run_id, *parts], ensure_ascii=False, separators=(",", ":"))
    return "viewer-" + hashlib.sha256(body.encode()).hexdigest()[:24]


# Validate only the existing public response contract and preserve exact released claim text.
def response(value):
    require(isinstance(value, dict) and set(value) == RESPONSE_KEYS, "Invalid response fields")
    request_id = string(value["request_id"], "request ID", nonempty=True)
    require(ID.fullmatch(request_id) is not None, "Invalid request ID")
    require(value["coverage"] == "not_established", "Unexpected coverage claim")
    require(isinstance(value["claims"], list) and len(value["claims"]) <= 32, "Invalid claims")
    claims = []
    for claim in value["claims"]:
        require(isinstance(claim, dict) and set(claim) == {"text", "evidence_refs"}, "Invalid claim fields")
        text = string(claim["text"], "released claim", nonempty=True)
        require(len(text) <= 2000, "Oversized claim")
        refs = claim["evidence_refs"]
        require(isinstance(refs, list) and refs and all(isinstance(r, str) and REF.fullmatch(r) for r in refs), "Invalid claim references")
        claims.append({"text": text, "evidence_refs": list(refs)})
    expected = "released" if claims else "unavailable"
    require(value["status"] == expected and value["answer"] == "\n".join(c["text"] for c in claims), "Response status/body does not match claims")
    return {"request_id": request_id, "status": expected, "answer": value["answer"], "claims": claims, "coverage": "not_established"}


# Reject duplicate JSON keys rather than silently overwriting a canonical message field.
def pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


# Parse untrusted saved JSON data without evaluating text or accepting non-finite constants.
def decode(text):
    # Reject nonstandard NaN and Infinity values in saved JSON.
    def invalid(_):
        raise ReplayError("Non-finite JSON number")
    try:
        return json.loads(text, object_pairs_hook=pairs_unique, parse_constant=invalid)
    except (ValueError, RecursionError) as exc:
        raise ReplayError("Malformed saved JSON") from exc


# Build the recipient view from canonical releases, keeping private projection and audit fields outside the result.
def build_view(run: dict, run_id: str, source_sha256: str) -> dict:
    require(isinstance(run, dict), "Run must be an object")
    require(isinstance(run_id, str) and ID.fullmatch(run_id), "Invalid explicit run ID")
    require(isinstance(source_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", source_sha256), "Invalid source digest")
    require(run.get("status") in {"running", "completed", "failed"}, "Unsupported saved run status")
    require(isinstance(run.get("config"), dict) and run["config"].get("protocol") in {"aos-artifact-requests-v1", "aos-artifact-requests-observed-v2"}, "Unsupported legacy protocol")
    # A versioned assessment must never be silently presented as an unobserved historical run.
    require(not any(key in run for key in ("conflicts", "conflict_audit", "conflict_detection")),
            "Unknown conflict extension needs an explicit versioned projection")
    observed = run["config"]["protocol"] == "aos-artifact-requests-observed-v2"
    require(("decision_observation" in run) == observed, "Decision observation does not match the run protocol")
    require(run["config"].get("mode") in {"governed", "prompt_only"}, "Unknown owner review treatment")
    require(not observed or run["config"]["mode"] == "governed", "Observed-v2 requires the declared governed owner treatment")
    if "system_version" in run["config"]:
        versions = {"observed-v2"} if observed else {"governed-v1", "integrated-v3"}
        require(run["config"]["system_version"] in versions, "System version does not match protocol")
        require(run["config"]["system_version"] != "integrated-v3" or run["config"]["mode"] == "governed",
                "Integrated-v3 requires the declared governed owner treatment")
    contract = response_contract_id(run["config"])
    projection = run.get("projection")
    require(isinstance(projection, dict), "Missing scenario projection")
    participants = projection.get("participants")
    require(isinstance(participants, list) and participants, "Missing participant list")
    people = []
    for p in participants:
        require(isinstance(p, dict) and isinstance(p.get("id"), str) and ID.fullmatch(p["id"]), "Invalid participant")
        people.append({"id": p["id"], "name": string(p.get("name"), "public participant name", nonempty=True)})
    ids = {p["id"] for p in people}
    require(len(ids) == len(people), "Duplicate participant")
    executor = projection.get("executor_id")
    require(executor in ids, "Unknown executor")
    recorded = run.get("responses")
    require(isinstance(recorded, list), "Missing responses list")
    by_request = {}
    owners = set()
    for item in recorded:
        require(isinstance(item, dict) and set(item) == {"owner_id", "response"} and item["owner_id"] in ids, "Invalid owner response")
        owner = item["owner_id"]
        require(owner not in owners, "Legacy run repeats an owner response")
        owners.add(owner)
        parsed = response(item["response"])
        require(parsed["request_id"] not in by_request, "Duplicate response request ID")
        by_request[parsed["request_id"]] = (owner, parsed)
    events = run.get("events")
    require(isinstance(events, list), "Missing event list")
    timeline = []
    requests = {}
    delivered = set()
    all_refs = set()
    for ordinal, event in enumerate(events):
        require(isinstance(event, dict) and set(event) == EVENT_KEYS, "Invalid canonical event fields")
        kind = event["kind"]
        require(kind in {"request", "response"} and event["delivery"] in {"attempted", "confirmed"}, "Unsupported event state")
        source, target, rid = event["sender_id"], event["recipient_id"], event["request_id"]
        require(source in ids and target in ids and source != target, "Unknown event participant")
        require(isinstance(rid, str) and ID.fullmatch(rid), "Invalid event request ID")
        timestamp = string(event["timestamp"], "recorded timestamp", nonempty=True)
        item = {"event_id": view_id(run_id, source_sha256, "event", ordinal), "id_origin": "assigned_by_viewer",
                "ordinal": ordinal, "kind": kind, "sender_id": source, "recipient_id": target,
                "request_id": rid, "delivery": event["delivery"], "timestamp": timestamp}
        if kind == "request":
            require(source == executor and rid not in requests, "Unexpected or duplicate request")
            requests[rid] = event
            item["body"] = string(event["body"], "request body", nonempty=True)
        else:
            require(target == executor and rid in requests and requests[rid]["recipient_id"] == source, "Response has no corresponding request")
            parsed = response(decode(string(event["body"], "response body")))
            require(parsed["request_id"] == rid and rid in by_request and by_request[rid] == (source, parsed), "Canonical response disagrees with recorded response")
            require(rid not in delivered, "Repeated delivered response")
            if event["delivery"] == "confirmed":
                require(requests[rid]["delivery"] == "confirmed", "Confirmed response has an unconfirmed request")
                delivered.add(rid)
                item["response"] = parsed
                all_refs.update(ref for c in parsed["claims"] for ref in c["evidence_refs"])
            else:
                # An attempted response is not proof its contents reached the requester.
                item["response"] = None
                item["visibility_note"] = "Delivery unconfirmed; not displayed as received evidence."
        timeline.append(item)
    local = []
    for rid, (owner, parsed) in by_request.items():
        if owner == executor:
            local.append({"event_id": view_id(run_id, source_sha256, "local", owner, rid),
                "id_origin": "assigned_by_viewer", "kind": "local_evidence", "owner_id": owner,
                "timestamp": None, "ordering": "not_recorded_relative_to_peer_messages", "response": parsed})
            all_refs.update(ref for c in parsed["claims"] for ref in c["evidence_refs"])
        else:
            require(rid in delivered or any(e["kind"] == "response" and e["request_id"] == rid and e["delivery"] == "attempted" for e in events), "Owner response lacks a matching transmission record")
    require(type(run.get("n_a2a")) is int and run["n_a2a"] == len(events), "A2A counter disagrees with canonical records")
    final = string(run.get("final_response", ""), "recorded final response")
    used = run.get("used_evidence_refs", [])
    require(isinstance(used, list) and all(isinstance(ref, str) and ref in all_refs for ref in used), "Final decision references evidence not recorded as received or local")
    if final:
        final_state = "completed" if run["status"] == "completed" else "recorded_before_run_finished"
    else:
        final_state = "not_available"
    require(run["status"] != "completed" or bool(final.strip()), "Completed run lacks a final response")
    insufficient = run.get("insufficient_evidence")
    require(insufficient is None or type(insufficient) is bool, "Invalid insufficient-evidence flag")
    result = {"schema": "aos-requester-replay-v1", "run_id": run_id, "source_sha256": source_sha256,
        "mode": "recorded_replay", "snapshot_status": run["status"],
        "source_kind": projection.get("benchmark") if projection.get("benchmark") in {"PiSAs", "synthetic"} else "unknown",
        "scenario": {"case_id": string(projection.get("case_id"), "case ID", nonempty=True),
                     "task": string(projection.get("task"), "public task", nonempty=True)},
        "executor_id": executor, "participants": people, "timeline": timeline, "local_evidence": local,
        "a2a_records": len(events), "confirmed_received_responses": len(delivered),
        "answer": {"state": final_state, "text": final, "used_evidence_refs": list(used), "insufficient_evidence": insufficient},
        "conflict_detection": {"status": "not_recorded", "note": "This legacy run has no explicit conflict-detection stage."},
        "evaluation": {"status": "not_loaded", "note": "No benchmark labels or judgments were loaded by this adapter."}}

    # Build stable identities from actual received claims, independent of growing snapshot hashes.
    evidence = []
    for rid, (owner, parsed) in by_request.items():
        if owner == executor or rid in delivered:
            for index, claim in enumerate(parsed["claims"], 1):
                evidence.append({"claim_id": f"{owner}:{rid}:c{index}", "owner_id": owner,
                                 "request_id": rid, **claim})
    result["evidence"] = evidence
    result["recorded_started"] = run.get("started") if isinstance(run.get("started"), str) else None
    result["recorded_finished"] = run.get("finished") if isinstance(run.get("finished"), str) else None
    duration = run.get("pipeline_time")
    result["recorded_seconds"] = duration if type(duration) in (int, float) and math.isfinite(duration) and duration >= 0 else None
    if contract is not None:
        result["response_contract_id"] = contract
    if "decision_observation" in run:
        result["conflict_detection"] = observation(run["decision_observation"], evidence)
    return result


# Validate the operator's case identity against native metadata without opening scenario files.
def native_case(pipeline, case):
    require(isinstance(case, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/scenario_[0-9]+", case),
            "A native replay needs an explicit family/scenario identity")
    family, scenario = case.split("/")
    config = pipeline.get("config")
    require(isinstance(config, dict) and config.get("system") == "decentralized",
            "Only the recorded PiSAs decentralized pipeline schema is supported")
    require(config.get("scenario") == scenario, "Native scenario differs from the selected case")
    if "task_family" in config:
        require(config["task_family"] == family, "Native task family differs from the selected case")
    case_id = family + "__" + scenario
    if "case_id" in config:
        require(config["case_id"] == case_id, "Native case ID differs from the selected case")
    if config.get("scenario_dir"):
        source = string(config["scenario_dir"], "native scenario directory").replace("\\", "/").rstrip("/")
        require(source.split("/")[-2:] == [family, scenario],
                "Native scenario directory differs from the selected case")
    return case_id


# Project native token-passing messages without manufacturing AOS releases, audits, or timestamps.
def build_native_view(pipeline, run_id, source_sha256, *, case):
    require(isinstance(pipeline, dict), "Native pipeline must be an object")
    require(isinstance(run_id, str) and ID.fullmatch(run_id), "Invalid explicit run ID")
    require(isinstance(source_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", source_sha256),
            "Invalid source digest")
    case_id = native_case(pipeline, case)
    task = string(pipeline.get("task_description"), "native public task", nonempty=True)
    final = string(pipeline.get("final_response", ""), "native final response")
    gathered = string(pipeline.get("gathered_info", ""), "native gathered summary")
    logs = pipeline.get("log")
    require(isinstance(logs, list) and len(logs) <= 10000, "Missing or oversized native message log")
    names = {}

    # Native logs identify people by name; viewer IDs are explicitly presentation-only.
    def identity(name):
        name = string(name, "native participant name", nonempty=True)
        if name not in names:
            names[name] = "native-" + hashlib.sha256(name.encode()).hexdigest()[:24]
        return names[name]

    inventory = pipeline.get("agent_contents", {})
    require(isinstance(inventory, dict), "Invalid native actor inventory")
    for name in inventory:
        identity(name)
    timeline, stages = [], []
    executors = set()
    for ordinal, event in enumerate(logs):
        require(isinstance(event, dict), "Invalid native log entry")
        kind = event.get("type")
        require(kind in {"sent", "received", "gathered", "final"}, "Unsupported native log entry type")
        sender_name = string(event.get("agent"), "native log author", nonempty=True)
        sender = identity(sender_name)
        body = string(event.get("message"), "native message")
        time = event.get("timestamp")
        require(time is None or isinstance(time, str), "Invalid native timestamp")
        if kind in {"gathered", "final"}:
            executors.add(sender)
            stages.append({"kind": kind, "source_log_index": ordinal, "sender_id": sender,
                           "body": body, "timestamp": time})
            continue
        target_name = string(event.get("to"), "native message recipient", nonempty=True)
        target = identity(target_name)
        require(sender != target, "Native message cannot be routed to its author")
        # Upstream 'received'.from_ is the previous speaker, not this message's author.
        if kind == "sent":
            require(event.get("from_") == sender_name, "Native sent-message author is inconsistent")
        if "peer_name" in event:
            require(event["peer_name"] == target_name, "Native peer recipient is inconsistent")
        previous = event.get("from_")
        require(previous is None or isinstance(previous, str), "Invalid native previous-speaker field")
        timeline.append({"event_id": view_id(run_id, source_sha256, "native", ordinal),
                         "id_origin": "assigned_by_viewer", "ordinal": len(timeline),
                         "source_log_index": ordinal, "source_type": kind,
                         "kind": "native_message", "sender_id": sender, "recipient_id": target,
                         "body": body, "timestamp": time, "request_id": None,
                         "delivery": "recorded_in_source", "source_from": previous})
    require(type(pipeline.get("n_a2a")) is int and pipeline["n_a2a"] == len(timeline),
            "Native A2A counter disagrees with recorded messages")
    require(len(executors) <= 1, "Native gathered and final stages name different executors")
    gathered_stages = [item for item in stages if item["kind"] == "gathered"]
    final_stages = [item for item in stages if item["kind"] == "final"]
    require(len(gathered_stages) <= 1 and len(final_stages) <= 1, "Repeated native terminal stages")
    if gathered_stages:
        require(gathered_stages[0]["body"] == gathered, "Native gathered summary differs from its log")
    if final_stages:
        require(final_stages[0]["body"] == final, "Native final answer differs from its log")
        require(gathered_stages and gathered_stages[0]["source_log_index"] < final_stages[0]["source_log_index"],
                "Native final answer lacks an earlier gathered stage")
        require(final_stages[0]["source_log_index"] == len(logs) - 1, "Native final stage is not terminal")
    explicit = pipeline.get("status")
    require(explicit is None or explicit in {"running", "completed", "failed", "interrupted"},
            "Unsupported native execution status")
    complete = bool(final.strip()) and bool(final_stages)
    require(explicit != "completed" or complete, "Completed native run lacks a recorded final stage")
    state = explicit or ("completed" if complete else "unknown")
    duration = pipeline.get("pipeline_time")
    duration = duration if type(duration) in (int, float) and math.isfinite(duration) and duration >= 0 else None
    degraded = pipeline.get("degraded", False)
    require(type(degraded) is bool, "Invalid native degraded flag")
    return {"schema": "aos-native-replay-v1", "run_id": run_id, "source_sha256": source_sha256,
            "mode": "native_replay", "replay_kind": "pisas_decentralized", "snapshot_status": state,
            "source_kind": "PiSAs", "scenario": {"case_id": case_id, "task": task},
            "executor_id": next(iter(executors), None),
            "participants": [{"id": value, "name": name, "id_origin": "assigned_by_viewer"}
                             for name, value in names.items()],
            "timeline": timeline, "local_evidence": [], "evidence": [], "a2a_records": len(timeline),
            "confirmed_received_responses": None,
            "native_gathering": {"text": gathered, "source": "pipeline.gathered_info",
                                 "source_log_index": gathered_stages[0]["source_log_index"] if gathered_stages else None,
                                 "note": "Native gathered summary; no AOS claim-release audit was recorded."},
            "answer": {"state": "completed" if state == "completed" and complete else
                       "recorded_before_run_finished" if final else "not_available", "text": final,
                       "used_evidence_refs": [], "insufficient_evidence": None},
            "conflict_detection": {"status": "not_recorded", "disagreements": [],
                                   "note": "This native pipeline did not record an AOS disagreement assessment."},
            "evaluation": {"status": "not_loaded"}, "degraded": degraded,
            "recorded_started": None, "recorded_finished": None, "recorded_seconds": duration,
            "recorded_exported": pipeline.get("ran_at_utc") if isinstance(pipeline.get("ran_at_utc"), str) else None,
            "source_note": "Recorded native messages, not AOS releases. Ordering follows the source log; "
                           "no missing timestamps, request IDs, or delivery receipts are invented."}


# Expose only the validated, source-linked observation that the decision model actually recorded.
def observation(value, evidence):
    require(isinstance(value, dict) and value.get("schema") == "aos-decision-observation-v1", "Unsupported decision observation")
    if value.get("status") in {"not_started", "started", "failed"}:
        return {"status": value["status"], "disagreements": [],
                "note": "No completed disagreement assessment has been recorded."}
    require(value.get("status") == "recorded", "Unsupported observation state")
    actions = {"resolved", "nonblocking", "unresolved_but_proceed", "insufficient_evidence"}
    require(value.get("decision_action") in actions | {"no_disagreement"}, "Invalid observation action")
    note = string(value.get("decision_note"), "decision note")
    require(len(note) <= 3000, "Oversized decision note")
    catalogue = {c["claim_id"]: c for c in evidence}
    items = value.get("disagreements")
    require(isinstance(items, list) and len(items) <= 32, "Invalid disagreement list")
    records = []
    ids = set()
    for item in items:
        require(isinstance(item, dict), "Invalid disagreement")
        did = string(item.get("id"), "disagreement ID", nonempty=True)
        require(ID.fullmatch(did) is not None and did not in ids, "Duplicate or invalid disagreement ID")
        ids.add(did)
        claim_ids = item.get("claim_ids")
        require(isinstance(claim_ids, list) and len(claim_ids) >= 2 and len(set(claim_ids)) == len(claim_ids)
                and all(cid in catalogue for cid in claim_ids), "Disagreement references unavailable claims")
        chosen = item.get("selected_claim_ids")
        require(isinstance(chosen, list) and len(set(chosen)) == len(chosen)
                and all(cid in claim_ids for cid in chosen), "Unknown resolution selection")
        require(item.get("kind") in {"factual_conflict", "criteria_tension", "interpretation_difference", "scoped_difference"}, "Invalid disagreement kind")
        require(item.get("action") in actions and type(item.get("material")) is bool, "Invalid disagreement state")
        if item["action"] == "resolved":
            require(chosen and len(chosen) < len(claim_ids), "Resolved disagreement needs a selected subset")
        refs = {ref for cid in claim_ids for ref in catalogue[cid]["evidence_refs"]}
        provided = item.get("evidence_refs", [])
        require(isinstance(provided, list) and all(ref in refs for ref in provided), "Unknown disagreement evidence reference")
        scope = string(item.get("scope"), "disagreement scope")
        reason = string(item.get("reason"), "disagreement reason", nonempty=True)
        require(len(scope) <= 2000 and len(reason) <= 3000, "Oversized disagreement text")
        records.append({"id": did, "kind": item["kind"], "claim_ids": list(claim_ids), "scope": scope,
                        "material": item["material"], "action": item["action"], "selected_claim_ids": list(chosen),
                        "reason": reason, "evidence_refs": list(provided)})
    return {"status": "recorded", "schema": value["schema"], "decision_action": value["decision_action"],
            "decision_note": note, "disagreements": records,
            "note": "Recorded model assessment; semantic correctness is not guaranteed."}
