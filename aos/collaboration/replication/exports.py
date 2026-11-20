# Export recorded replication results without changing the demo workspace.

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid
import zipfile
from ..demo.evaluation import attach_evaluation, attach_native_evaluation
from ..demo.view import build_native_view, build_view, decode


# Hash exact artifact bytes for stage bindings and archive inventories.
def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Hash a plan independently of its presentation whitespace.
def plan_hash(plan):
    raw = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Keep every selected path inside its execution directory without following links.
def confined(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or not part.parts or any(item in {".", ".."} for item in part.parts):
        raise ValueError("Export paths must stay inside the execution directory")
    path = root.joinpath(*part.parts)
    if path.resolve() != path or not path.resolve().is_relative_to(root):
        raise ValueError("Export paths cannot follow symbolic links")
    return path


# Read bounded canonical objects while rejecting duplicate fields and nonfinite numbers.
def read_object(path):
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("An export source exceeds the size limit")
    value = decode(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("An export source must contain a JSON object")
    return value


# Replace generated summaries atomically while preserving their source artifacts.
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# Exclude credential-bearing metadata without rewriting exact scientific records.
def reject_credentials(value):
    fields = {"api_key", "token", "request_token", "admin_token", "credentials", "credential", "authorization"}
    for item in (value, value.get("config"), value.get("request")):
        if not isinstance(item, dict):
            continue
        if any(str(key).lower() in fields for key in item):
            raise ValueError("Credential-bearing metadata cannot be exported")
        headers = item.get("headers")
        if isinstance(headers, dict) and any(str(key).lower() == "authorization" for key in headers):
            raise ValueError("Credential-bearing headers cannot be exported")


# Select execution rows by the plan's exact stage identities, including repetition index.
def execution_rows(plan, status, run):
    ids = {stage["id"] for stage in plan["stages"] if stage["path"] == run["path"]}
    return [row for row in status.get("records", []) if row.get("id") in ids]


# Verify the accepted judgment against both its selected pipeline and saved result receipt.
def evaluation(root, run, view, metadata, execution):
    directory = confined(root, run["path"])
    states = {row["stage"]: row.get("status") for row in execution}
    empty = {"status": "not_loaded", "note": "No accepted matching judgment is available."}
    if states.get("generate") != "ok" or states.get("judge") != "ok":
        return empty
    receipt_path = confined(root, run["path"] + "/judge.json")
    pipeline_path = confined(root, run["path"] + "/pipeline.json")
    judgment_path = confined(root, run["path"] + "/judgment.json")
    if not all(path.is_file() for path in (receipt_path, pipeline_path, judgment_path)):
        raise ValueError("An accepted judgment is missing its receipt or output")
    receipt = read_object(receipt_path)
    if (receipt.get("case") != run["case"] or receipt.get("arm") != run["arm"]
            or receipt.get("run_index", 0) != run["run_index"] or receipt.get("stage") != "judge"):
        raise ValueError("Judgment receipt names a different replication slot")
    if receipt.get("status") != "ok" or receipt.get("evaluated_pipeline_sha256") != file_hash(pipeline_path):
        raise ValueError("Judgment receipt does not bind the selected pipeline")
    identities = receipt.get("artifact_sha256", {})
    if identities.get(judgment_path.relative_to(root).as_posix()) != file_hash(judgment_path):
        raise ValueError("Judgment receipt does not bind the selected evaluation")
    if run["arm"] == "native":
        result = attach_native_evaluation(judgment_path, pipeline_path, view, metadata, case=run["case"])
    else:
        result = attach_evaluation(judgment_path, pipeline_path, view, metadata)
    result.update(binding="verified_stage_receipt", receipt_sha256=file_hash(receipt_path),
                  binding_note="The successful judgment receipt binds this exact pipeline and evaluation.")
    return result


# Produce the existing public replay format from canonical results without opening a demo store.
def replay_record(root, plan, run, execution):
    directory = confined(root, run["path"])
    native = run["arm"] == "native"
    source = confined(root, run["path"] + ("/pipeline.json" if native else "/actor/run.json"))
    if not source.is_file():
        return None
    raw = read_object(source)
    digest = file_hash(source)
    run_id = "run-" + digest[:32]
    view = build_native_view(raw, run_id, digest, case=run["case"]) if native else build_view(raw, run_id, digest)
    if view["scenario"]["case_id"] != run["case"].replace("/", "__"):
        raise ValueError("Recorded case differs from the planned replication slot")
    config = raw["config"]
    model = config["agent_llm"] if native else config["model"]
    if model != run["model"] or config.get("run_index", 0) != run["run_index"]:
        raise ValueError("Recorded model or repetition differs from the planned replication slot")
    version = "native-decentralized" if native else config.get("system_version", "governed-v1")
    expected = {"native": "native-decentralized", "integrated": "integrated-v3", "governed-v1": "governed-v1"}[run["arm"]]
    if version != expected:
        raise ValueError("Recorded system differs from the planned replication slot")
    states = {row["stage"]: row for row in execution}
    generation = states.get("generate", {})
    state = view["snapshot_status"] if generation.get("status") == "ok" else generation.get("status", "unknown")
    record = {"run_id": run_id, "title": f"Replication · {plan['phase']} · {run['case']} · {version} · run {run['run_index']}",
              "case_id": view["scenario"]["case_id"], "version": version, "model": model,
              "run_index": run["run_index"], "mode": view["mode"], "status": state,
              "created": view.get("recorded_started"), "updated": view.get("recorded_finished"),
              "source_sha256": digest, "backend": "model", "cohort": plan["phase"],
              "generation_stage_status": generation.get("status", "not_started")}
    receipt = confined(root, run["path"] + "/generate.json")
    if receipt.is_file():
        record["generation_receipt_sha256"] = file_hash(receipt)
        saved = read_object(receipt)
        if generation.get("status") == "ok" and (saved.get("status") != "ok"
                or saved.get("artifact_sha256", {}).get(source.relative_to(root).as_posix()) != digest):
            raise ValueError("Generation receipt does not bind the selected replay source")
        attempt_path = confined(root, run["path"] + "/generate-attempt.json")
        attempt = read_object(attempt_path) if attempt_path.is_file() else {}
        record["created"] = record["created"] or attempt.get("started")
        record["updated"] = record["updated"] or saved.get("finished")
    elif generation.get("status") == "ok":
        raise ValueError("A completed generation is missing its result receipt")
    if view.get("response_contract_id"):
        record["response_contract_id"] = view["response_contract_id"]
    if not native and bool(run.get("response_contract")) != bool(view.get("response_contract_id")):
        raise ValueError("Recorded response contract differs from the planned treatment")
    judged = evaluation(root, run, view, record, execution)
    record["evaluation_status"] = judged["status"]
    if generation.get("status") != "ok":
        record["warning"] = "Generation did not complete successfully; recorded text is retained without an accepted completion or score."
    return {"schema": "aos-collaboration-replay-seed-v1", "record": record, "view": view, "evaluation": judged}


# Collect private run artifacts and explicitly named owner audits for the common readout.
def readout_record(root, run, execution):
    directory = confined(root, run["path"])
    result = {**run, "execution": execution}
    for name, key in (("pipeline.json", "pipeline"), ("judgment.json", "judgment"), ("actor/run.json", "actor_run")):
        path = confined(root, run["path"] + "/" + name)
        if path.is_file():
            result[key + "_sha256"] = file_hash(path)
            try:
                result[key] = read_object(path)
            except (ValueError, UnicodeError):
                result[key + "_parse_error"] = "The saved artifact is unreadable; its exact bytes remain in the analysis archive."
            else:
                reject_credentials(result[key])
    audits = []
    for person in result.get("actor_run", {}).get("projection", {}).get("participants", []):
        owner = person.get("id", "")
        if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", owner):
            raise ValueError("Invalid recorded owner identity")
        for path in sorted((directory / "actor/owners" / owner / "audit").glob("req-*.json")):
            path = confined(root, path.relative_to(root).as_posix())
            audit = read_object(path)
            reject_credentials(audit)
            audits.append({"owner_id": owner, "sha256": file_hash(path), "record": audit})
    if audits:
        result["owner_audits"] = audits
    return result


# Select only declared outputs, receipts and audits rather than archiving owner credentials.
def archive_sources(root, plan, replay_paths):
    selected = set()

    # Add a confined regular file only when its known export surface exists.
    def add(relative):
        path = confined(root, relative)
        if path.is_file():
            selected.add(path)

    for name in ("plan.json", "source-identities.json", "status.json", "readout.json", "replay.json", "progress.log"):
        add(name)
    for relative in replay_paths:
        add(relative)
    for run in plan["runs"]:
        directory = confined(root, run["path"])
        for name in ("generate-attempt.json", "generate.json", "judge-attempt.json", "judge.json",
                     "pipeline.json", "judgment.json", "actor/run.json"):
            add(run["path"] + "/" + name)
        for stage in ("generation", "judging"):
            for name in ("profile.json", "upstream.log"):
                add(run["path"] + "/" + stage + "/" + name)
            for path in sorted((directory / stage / "calls").glob("*.json")):
                if re.fullmatch(r"[0-9]+\.json", path.name):
                    add(path.relative_to(root).as_posix())
            for path in sorted((directory / stage / "response-scopes").glob("*.json")):
                if re.fullmatch(r"[0-9]+\.json", path.name):
                    add(path.relative_to(root).as_posix())
        for path in sorted((directory / "actor/owners").glob("*/audit/req-*.json")):
            add(path.relative_to(root).as_posix())
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


# Write current summaries and a byte-indexed private archive without invoking providers.
def export_bundle(root, plan, status):
    root = Path(root).expanduser().resolve()
    if plan_hash(read_object(confined(root, "plan.json"))) != plan_hash(plan) or status.get("plan_sha256") != plan_hash(plan):
        raise ValueError("The export plan differs from the recorded execution plan")
    created = datetime.now(timezone.utc).isoformat()
    common = {"created": created, "phase": plan["phase"], "profile": plan.get("profile", "fixed-batches-v1"),
              "actor_model": plan["actor_model"], "plan_sha256": plan_hash(plan), "provider_requests": 0}
    records, replays, replay_paths = [], [], []
    for run in plan["runs"]:
        execution = execution_rows(plan, status, run)
        row = readout_record(root, run, execution)
        if "judging_profile" in plan:
            row["judging_profile"] = plan["judging_profile"]
        public = {"case": run["case"], "system": run["arm"], "run_index": run["run_index"], "replay": "missing"}
        try:
            snapshot = replay_record(root, plan, run, execution)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            public.update(replay="failed", error_type=type(exc).__name__, reason=str(exc))
            row["replay_error"] = str(exc)
        else:
            if snapshot is not None:
                record = snapshot["record"]
                relative = "replays/" + record["run_id"] + ".json"
                write_json(confined(root, relative), snapshot)
                replay_paths.append(relative)
                public.update(replay="exported", path=relative, run_id=record["run_id"], status=record["status"],
                              version=record["version"], evaluation=record["evaluation_status"], source_sha256=record["source_sha256"])
                if snapshot["evaluation"]["status"] == "attached":
                    row["official_metrics"] = snapshot["evaluation"]["metrics"]
        if "judgment" in row and "official_metrics" not in row:
            row["official_metrics_unavailable"] = "No successful matching judgment was accepted; raw evaluator output is retained."
        records.append(row)
        replays.append(public)
    readout = {"schema": "aos-replication-readout-v1", **common, "specification": plan["specification"],
               "status": status, "records": records,
               "privacy": "Private experimental analysis containing owner evidence and evaluator material; do not pass it to actors.",
               "interpretation": "Separate execution health, decision correctness, evidence coverage and disclosure flags. Repetitions are reported individually."}
    available = sum(row["replay"] == "exported" for row in replays)
    replay = {"schema": "aos-replication-replay-export-v1", **common, "plan_file_sha256": file_hash(root / "plan.json"),
              "scope": "Only the fixed batch's declared recorded slots", "runs": replays, "planned_runs": len(replays),
              "available_runs": available, "all_planned_runs_available": available == len(replays), "gui_imported": False}
    for name, value in (("status.json", status), ("readout.json", readout), ("replay.json", replay)):
        reject_credentials(value)
        write_json(confined(root, name), value)
    target = confined(root, "analysis.zip")
    temporary = target.with_name("analysis." + uuid.uuid4().hex + ".tmp")
    index = {"schema": "aos-replication-analysis-bundle-v1", "operator_only": True, **common,
             "plan_file_sha256": file_hash(root / "plan.json"), "source_sha256": plan.get("source_sha256", {}),
             "inputs": plan.get("inputs", []), "files": [],
             "content_note": "Contains exact private actor, evaluator and call records; public replay projections are separate.",
             "excluded": [".env", "service manifests", "endpoints", "owner workspaces", "raw scenario input bundles"]}
    try:
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in archive_sources(root, plan, replay_paths):
                    raw = path.read_bytes()
                    if path.suffix == ".json":
                        try:
                            value = decode(raw)
                        except (ValueError, UnicodeError):
                            if path.name not in {"pipeline.json", "judgment.json", "run.json"}:
                                raise
                        else:
                            if isinstance(value, dict):
                                reject_credentials(value)
                    relative = path.relative_to(root).as_posix()
                    archive.writestr(relative, raw)
                    index["files"].append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)})
                archive.writestr("bundle-index.json", json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"schema": "aos-replication-export-v1", "phase": plan["phase"], "provider_requests": 0,
            "status": str(root / "status.json"), "readout": str(root / "readout.json"), "replay": str(root / "replay.json"),
            "analysis": str(target), "analysis_sha256": file_hash(target), "available_runs": available,
            "planned_runs": len(replays), "gui_imported": False}
