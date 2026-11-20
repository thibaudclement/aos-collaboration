# Execute fixed batches with separate actor inputs, evaluator inputs, and immutable stage receipts.

from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from ...llm import load_env
from ..common import write_json
from ..demo.bootstrap import bundled_data
from ..experiment import prepare_experiment
from ..integrated import contract, contract_runtime, runtime
from ..pisas import assert_operator_path, load_scenario
from . import batches, judge, native
from .exports import export_bundle
from .provider import Budget, MeteredModel, OPENROUTER_URL, ProviderError, Transport, amount, digest, write_record


# Report invalid state without replacing or silently rerunning an earlier experiment.
class ReplicationError(RuntimeError):
    pass


# Record timezone-aware timestamps independently of terminal locale.
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Hash exact file bytes for source and output bindings.
def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Read one canonical object without accepting nonfinite JSON measurements.
def read_object(path: Path) -> dict:
    from ..demo.view import decode
    value = decode(path.read_bytes())
    if not isinstance(value, dict):
        raise ReplicationError("Expected a JSON object: " + path.name)
    return value


# Keep output paths separate from bundled data and existing AOS workspaces.
def output_root(path: str | Path) -> Path:
    root = assert_operator_path(path)
    package = Path(__file__).resolve().parents[2]
    data = bundled_data().resolve()
    if (root.is_relative_to(package) or root.is_relative_to(data)
            or ".aos-collaboration" in root.parts):
        raise ReplicationError("Choose an experiment directory outside the demo workspace and source files")
    return root


# Confine recorded paths to their batch directory without following symbolic links.
def child(root: Path, relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or not part.parts or any(item in {".", ".."} for item in part.parts):
        raise ReplicationError("Invalid relative experiment path")
    path = root / part
    if path.resolve() != path or not path.resolve().is_relative_to(root):
        raise ReplicationError("Experiment files cannot be redirected outside their directory")
    return path


# Prevent concurrent commands from dispatching the same paid stage.
@contextmanager
def batch_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = child(root, ".batch.lock")
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ReplicationError("Another process owns this batch directory") from None
        yield


# Freeze the executing Python implementation and selected bundled source inputs.
def source_fingerprint(batch: dict) -> dict:
    package = Path(__file__).resolve().parents[2]
    identities = {"aos/" + path.relative_to(package).as_posix(): file_hash(path)
                  for path in sorted(package.rglob("*.py"))}
    data = bundled_data()
    for relative in ("replication.json", *("scenarios/" + case + "/scenario.json" for case in batch["cases"])):
        identities["data/" + relative] = file_hash(data / relative)
    native.source_fingerprint()
    return identities


# Place each generation immediately before its official judgment in the fixed run order.
def planned_stages(batch: dict, runs: list[dict]) -> list[dict]:
    stages = []
    for run in runs:
        for kind in ("generate", "judge"):
            stage = {**run, "id": f"{run['case']}:{run['arm']}:run{run['run_index']}:{kind}", "stage": kind}
            if kind == "judge":
                stage.update(maximum_calls=batch["judge_maximum_calls"], base_url=OPENROUTER_URL, key_env="API_KEY")
            stages.append(stage)
    return stages


# Prepare a fresh offline plan without creating a provider or opening the demo store.
def prepare_plan(name: str, root: Path) -> dict:
    batch = batches.get_batch(name)
    if any(path.name != ".batch.lock" for path in root.iterdir()):
        raise ReplicationError("The output directory contains files without a valid plan; choose a fresh directory")
    sources = source_fingerprint(batch)
    inputs = batches.prepare_inputs(batch, root)
    runs = batches.planned_runs(batch, inputs)
    plan = {"schema": "aos-replication-plan-v1", "phase": name, "created": now(),
            "profile": "fixed-batches-v1", "actor_model": batch["actor_model"],
            "specification": batch, "judging_profile": judge.specification(batch["judge_profile"]),
            "source_sha256": sources, "inputs": inputs, "runs": runs,
            "stages": planned_stages(batch, runs), "gui_imported": False}
    write_record(root / "plan.json", plan)
    write_record(root / "source-identities.json", {"schema": "aos-replication-source-identities-v1",
                 "plan_sha256": digest(plan), "plan_file_sha256": file_hash(root / "plan.json"),
                 "source_sha256": sources, "upstream_revision": native.UPSTREAM_REVISION})
    return plan


# Validate fixed settings and all frozen inputs before continuing or exporting a batch.
def load_plan(name: str, root: Path) -> dict:
    plan = read_object(child(root, "plan.json"))
    identities = read_object(child(root, "source-identities.json"))
    batch = batches.get_batch(name)
    if (plan.get("schema") != "aos-replication-plan-v1" or plan.get("phase") != name
            or digest(plan.get("specification")) != digest(batch)
            or plan.get("judging_profile") != judge.specification(batch["judge_profile"])
            or identities.get("plan_sha256") != digest(plan)
            or identities.get("plan_file_sha256") != file_hash(root / "plan.json")
            or plan.get("source_sha256") != source_fingerprint(batch)
            or identities.get("source_sha256") != plan["source_sha256"]):
        raise ReplicationError("The saved plan or source files changed; do not continue this batch")
    if [row.get("case") for row in plan["inputs"]] != list(batch["cases"]):
        raise ReplicationError("The selected input inventory changed")
    for row in plan["inputs"]:
        if (row.get("actor_scenario") != "actor-inputs/" + row["case"] + "/scenario.json"
                or row.get("judge_input") != "inputs/" + row["case"]):
            raise ReplicationError("A prepared input path changed")
        actor = child(root, row["actor_scenario"])
        if file_hash(actor) != row["actor_sha256"]:
            raise ReplicationError("A frozen actor scenario changed")
        for filename, expected in row["input_files"].items():
            if file_hash(child(root, row["judge_input"] + "/" + filename)) != expected:
                raise ReplicationError("A frozen evaluator input changed")
        if file_hash(child(root, row["judge_input"] + "/scenario.json")) != row["actor_sha256"]:
            raise ReplicationError("Actor and evaluator scenario bytes differ")
    runs = batches.planned_runs(batch, plan["inputs"])
    if plan["runs"] != runs or plan["stages"] != planned_stages(batch, runs):
        raise ReplicationError("The run schedule or request limits changed")
    return plan


# Read only the serial provider calls belonging to one planned stage.
def call_records(root: Path, stage: dict) -> tuple[list[dict], dict]:
    kind = "generation" if stage["stage"] == "generate" else "judging"
    folder = child(root, stage["path"] + "/" + kind + "/calls")
    paths = sorted(folder.glob("*.json"), key=lambda path: int(path.stem)) if folder.exists() else []
    rows, identities = [], {}
    for number, path in enumerate(paths, 1):
        if path.name != f"{number:03d}.json":
            raise ReplicationError("The stage call sequence is missing or renamed")
        checked = child(root, path.relative_to(root).as_posix())
        record = read_object(checked)
        if record.get("call") != number:
            raise ReplicationError("A call receipt changed identity")
        rows.append(record)
        identities[path.relative_to(root).as_posix()] = file_hash(checked)
    if len(rows) > stage["maximum_calls"]:
        raise ReplicationError("The recorded stage exceeded its call ceiling")
    return rows, identities


# Keep unreported provider prices distinct from incomplete or failed paid requests.
def call_totals(calls: list[dict]) -> dict:
    return {"model_calls": len(calls),
            "known_response_cost_usd": round(sum(amount(row["provider_cost_usd"], "saved response cost")
                for row in calls if row.get("provider_cost_usd") is not None), 12),
            "unknown_cost_calls": sum(row.get("provider_cost_usd") is None for row in calls),
            "unreported_cost_calls": sum(row.get("cost_status") == "not_reported_by_provider" for row in calls)}


# Reconstruct stage state from attempt receipts and exact output identities.
def stage_state(root: Path, plan: dict, stage: dict) -> dict:
    directory = child(root, stage["path"])
    attempt_path = directory / (stage["stage"] + "-attempt.json")
    result_path = directory / (stage["stage"] + ".json")
    output = directory / ("pipeline.json" if stage["stage"] == "generate" else "judgment.json")
    calls, call_ids = call_records(root, stage)
    state = {key: stage[key] for key in ("id", "case", "arm", "run_index", "stage")}
    state.update(status="not_started", **call_totals(calls))
    if not attempt_path.exists():
        if (result_path.exists() or output.exists() or calls
                or stage["stage"] == "generate" and (directory / "actor/run.json").exists()):
            raise ReplicationError("A stage has outputs without its attempt receipt")
        return state
    attempt = read_object(child(root, attempt_path.relative_to(root).as_posix()))
    if attempt.get("plan_sha256") != digest(plan) or attempt.get("stage") != stage:
        raise ReplicationError("A stage attempt no longer matches the frozen plan")
    if not result_path.exists():
        state["status"] = "interrupted_or_running"
        return state
    result = read_object(child(root, result_path.relative_to(root).as_posix()))
    if (result.get("attempt_sha256") != digest(attempt) or result.get("call_receipt_sha256") != call_ids
            or result.get("status") not in {"ok", "failed", "interrupted"}):
        raise ReplicationError("A stage result or its call receipts changed")
    for relative, expected in result.get("artifact_sha256", {}).items():
        if file_hash(child(root, relative)) != expected:
            raise ReplicationError("A recorded stage artifact changed")
    if result["status"] == "ok":
        if not calls or any(row.get("status") != "ok" for row in calls):
            raise ReplicationError("A successful stage lacks successful provider-call receipts")
        if any(row.get("provider_cost_usd") is None and not (
                stage["key_env"] == "GOOGLE_API_KEY" and row.get("cost_status") == "not_reported_by_provider")
                for row in calls):
            raise ReplicationError("A successful stage has an unexplained missing provider charge")
        if result.get("artifact_sha256", {}).get(output.relative_to(root).as_posix()) != file_hash(output):
            raise ReplicationError("A successful stage lacks a bound canonical output")
        if stage["stage"] == "judge" and result.get("evaluated_pipeline_sha256") != file_hash(directory / "pipeline.json"):
            raise ReplicationError("The accepted judgment belongs to a different pipeline")
    state.update(status=result["status"], elapsed_seconds=result.get("elapsed_seconds"),
                 error_type=result.get("error_type"))
    return state


# Select only the first never-started stage after an intact successful prefix.
def status_from_plan(root: Path, plan: dict) -> dict:
    records = [stage_state(root, plan, stage) for stage in plan["stages"]]
    counts = Counter(row["status"] for row in records)
    pending = next((index for index, row in enumerate(records) if row["status"] != "ok"), None)
    stop = None
    next_stage = None
    if pending is not None:
        row = records[pending]
        if row["status"] != "not_started":
            stop = "A stage failed or was interrupted; its paid calls will not be retried automatically."
        elif any(item["status"] != "not_started" for item in records[pending + 1:]):
            raise ReplicationError("The recorded stages no longer form the fixed execution prefix")
        else:
            next_stage = plan["stages"][pending]
    return {"schema": "aos-replication-status-v1", "phase": plan["phase"], "profile": plan["profile"],
            "actor_model": plan["actor_model"], "plan_sha256": digest(plan),
            "runs": len(plan["runs"]), "stages": len(records), "statuses": dict(counts),
            "model_calls": sum(row["model_calls"] for row in records),
            "known_response_cost_usd": round(sum(row["known_response_cost_usd"] for row in records), 12),
            "unknown_cost_calls": sum(row["unknown_cost_calls"] for row in records),
            "unreported_cost_calls": sum(row["unreported_cost_calls"] for row in records),
            "stop": stop, "next": next_stage, "records": records, "provider_requests": 0,
            "gui_imported": False}


# Copy an exact canonical output without replacing an earlier result.
def copy_output(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with source.open("rb") as source_file, destination.open("xb") as target:
        shutil.copyfileobj(source_file, target)
    destination.chmod(0o600)


# Execute one recorded treatment through the existing AOS or pinned PiSAs runtime.
def execute_stage(root: Path, plan: dict, stage: dict, budget, progress) -> dict:
    directory = child(root, stage["path"])
    selected = next(row for row in plan["inputs"] if row["case"] == stage["case"])
    inputs = child(root, selected["judge_input"])
    batch = batches.get_batch(plan["phase"])
    if stage["stage"] == "judge":
        pipeline_path = directory / "pipeline.json"
        output = judge.evaluate(inputs, read_object(pipeline_path), directory / "judging", budget,
                                actor_model=stage["model"], profile=batch["judge_profile"],
                                max_calls=stage["maximum_calls"], progress=progress)
        copy_output(Path(output["upstream_output"]), directory / "judgment.json")
        return {"health": output["health"], "usage": output["usage"],
                "evaluated_pipeline_sha256": file_hash(pipeline_path),
                "judging_profile": judge.specification(batch["judge_profile"])}
    if stage["arm"] == "native":
        output = native.generate(inputs, directory / "generation", budget,
                                 actor_model=stage["model"], max_calls=stage["maximum_calls"], progress=progress)
        pipeline = deepcopy(output["pipeline"])
        pipeline["config"]["run_index"] = stage["run_index"]
        write_record(directory / "pipeline.json", pipeline)
        return {"health": output["health"], "usage": output["usage"]}
    projection = load_scenario(child(root, selected["actor_scenario"]))
    if len(projection["participants"]) != selected["owner_count"]:
        raise ReplicationError("Actor participants differ from the frozen plan")
    manifest = prepare_experiment(projection, directory / "actor", model=stage["model"], mode="governed",
                                  top_k=None if stage["arm"] == "integrated" else 8, run_index=stage["run_index"])
    manifest["config"].update(system_version=stage["version"], integration_treatment=stage["arm"])
    if stage["response_contract"]:
        manifest["config"]["response_contract"] = contract.specification()
    write_json(directory / "actor/manifest.json", manifest)
    model = MeteredModel(budget, directory / "generation", stage["model"],
                         max_calls=stage["maximum_calls"], progress=progress)
    engine = contract_runtime if stage["response_contract"] else runtime
    recorded = engine.run(directory / "actor", model, treatment=stage["arm"], progress=progress)
    exported = batches.export_pisas(recorded, directory / "aos-export")
    pipeline = read_object(exported)
    pipeline["config"].update(aos_system_version=stage["version"], integration_treatment=stage["arm"])
    if stage["response_contract"]:
        pipeline["config"]["response_contract"] = contract.specification()
    write_record(directory / "pipeline.json", pipeline)
    return {"health": {"status": recorded["status"], "degraded": bool(recorded.get("degraded", False))},
            "usage": deepcopy(model.usage)}


# Preserve successful, failed, and interrupted stage attempts before any later dispatch.
def attempt_stage(root: Path, plan: dict, stage: dict, budget, progress) -> dict:
    directory = child(root, stage["path"])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    attempt = {"schema": "aos-replication-stage-attempt-v1", "started": now(),
               "plan_sha256": digest(plan), "stage": stage,
               "inference_retry_policy": "No transport retry; fixed judge response loops only."}
    write_record(directory / (stage["stage"] + "-attempt.json"), attempt)
    result = {"schema": "aos-replication-stage-result-v1", "attempt_sha256": digest(attempt),
              **{key: stage[key] for key in ("case", "arm", "run_index", "stage")}}
    started = time.monotonic()
    interrupted = None
    try:
        result.update(execute_stage(root, plan, stage, budget, progress))
        result["status"] = "ok"
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                      error_type=type(exc).__name__)
        if hasattr(exc, "usage"):
            result["usage"] = exc.usage
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupted = exc
    finally:
        if stage["stage"] == "judge" or stage["arm"] == "native":
            kind = "judging" if stage["stage"] == "judge" else "generation"
            prefix = "evaluation_" if stage["stage"] == "judge" else "pipeline_"
            scenario = stage["case"].split("/")[1]
            saved = directory / kind / "upstream" / scenario / (prefix + scenario + ".json")
            target = directory / ("judgment.json" if stage["stage"] == "judge" else "pipeline.json")
            if saved.is_file() and not target.exists():
                copy_output(saved, target)
        calls, call_ids = call_records(root, stage)
        result.update(finished=now(), elapsed_seconds=round(time.monotonic() - started, 3),
                      call_receipt_sha256=call_ids, **call_totals(calls))
        names = ["judgment.json"] if stage["stage"] == "judge" else ["pipeline.json", "actor/run.json"]
        result["artifact_sha256"] = {str((directory / name).relative_to(root)): file_hash(directory / name)
                                    for name in names if (directory / name).is_file()}
        if stage["stage"] == "generate":
            for audit in sorted((directory / "actor/owners").glob("*/audit/*.json")):
                result["artifact_sha256"][audit.relative_to(root).as_posix()] = file_hash(audit)
        write_record(directory / (stage["stage"] + ".json"), result)
    if interrupted is not None:
        raise interrupted
    return result


# Resolve required credentials locally while keeping all batch models and endpoints fixed.
def credentials(plan: dict) -> dict[str, str]:
    load_env()
    names = {stage["key_env"] for stage in plan["stages"]}
    values = {}
    for name in names:
        value = os.environ.get(name, "").strip()
        if name == "API_KEY":
            value = os.environ.get("OPENROUTER_API_KEY", "").strip() or value
        if not value or value.lower().startswith(("your-", "your_", "replace-", "replace_", "<")):
            raise ReplicationError("Set " + name + " before paid execution; this batch uses fixed provider settings")
        values[name] = value
    return values


# Preview a fixed batch offline or run its intact remaining prefix with explicit paid consent.
def run(name: str, output: str | Path, *, execute: bool = False, progress=print, transport_factory=Transport) -> dict:
    root = output_root(output)
    with batch_lock(root):
        plan = load_plan(name, root) if (root / "plan.json").exists() else prepare_plan(name, root)
        status = status_from_plan(root, plan)
        if not execute or not status["next"] or status["stop"]:
            return {"executed": False, "preview": not execute, "output": str(root),
                    "status": status, "exports": export_bundle(root, plan, status)}
        progress("PAID EXECUTION: this batch invokes actor models and official judges; charges are not capped by this CLI.")
        keys = credentials(plan)
        progress(f"Starting {name}: {len(plan['runs'])} runs, {len(plan['stages'])} stages; output={root}")
        with (root / "progress.log").open("a", encoding="utf-8") as log:
            # Send each operational update to the terminal and a durable local transcript.
            def notice(message):
                line = f"[{now()}] {message}"
                log.write(line + "\n")
                log.flush()
                progress(line)

            execution_error = None
            try:
                for index, stage in enumerate(plan["stages"], 1):
                    current = stage_state(root, plan, stage)
                    if current["status"] == "ok":
                        continue
                    if current["status"] != "not_started":
                        break
                    # Recheck frozen code and inputs before dispatching another paid stage.
                    if digest(load_plan(name, root)) != digest(plan):
                        raise ReplicationError("The execution plan changed during this batch")
                    batch = batches.get_batch(name)
                    timeout = (batch["judge_timeout_seconds"] if stage["stage"] == "judge" else
                               batch["native_timeout_seconds"] if stage["arm"] == "native" else batch["aos_timeout_seconds"])
                    transport = transport_factory(keys[stage["key_env"]], base_url=stage["base_url"], timeout=timeout)
                    budget = Budget(transport)
                    notice(f"[{index}/{len(plan['stages'])}] {name} | {stage['case']} | {stage['version']} | "
                           f"run {stage['run_index']} | {stage['stage']}")
                    result = attempt_stage(root, plan, stage, budget, notice)
                    notice(f"Stage {result['status']}; {result['elapsed_seconds']:.1f}s; "
                           f"calls={result['model_calls']}; known charges=${result['known_response_cost_usd']:.6f}")
                    write_record(root / "status.json", status_from_plan(root, plan), replace=True)
                    if result["status"] != "ok":
                        break
            except BaseException as exc:
                execution_error = exc
                raise
            finally:
                try:
                    status = status_from_plan(root, plan)
                    exported = export_bundle(root, plan, status)
                except Exception as exc:
                    if execution_error is None:
                        raise
                    progress(f"Export did not finish ({type(exc).__name__}); canonical stage files remain in {root}.")
        return {"executed": True, "preview": False, "output": str(root), "status": status, "exports": exported}


# Inspect an existing batch without constructing a provider or invoking a model.
def status(name: str, output: str | Path) -> dict:
    root = output_root(output)
    return status_from_plan(root, load_plan(name, root))


# Rebuild the common offline exports while leaving canonical run artifacts unchanged.
def export(name: str, output: str | Path) -> dict:
    root = output_root(output)
    with batch_lock(root):
        plan = load_plan(name, root)
        return export_bundle(root, plan, status_from_plan(root, plan))
