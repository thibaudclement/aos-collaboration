# Persistent local conversation catalogue, replay records, and bounded live launches.

from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import traceback
import uuid
from ..experiment import DEFAULT_MODEL, demo_projection
from ..pisas import assert_operator_path, load_scenario
from .evaluation import attach_evaluation, attach_native_evaluation
from .view import build_view, build_native_view, decode, ReplayError, RESPONSE_CONTRACT_ID

VERSIONS = ("governed-v1", "observed-v2", "integrated-v3")
RUN_ID = re.compile(r"run-[0-9a-f]{32}\Z")
LAUNCH_ID = re.compile(r"[A-Za-z0-9_-]{8,100}\Z")
ACTIVE = {"queued", "running"}


# Carry a safe operator-facing error and indicate whether an execution may have started.
class StoreError(ValueError):
    # Retain HTTP semantics without returning private exception details.
    def __init__(self, message, status=400, *, outcome_unknown=False):
        super().__init__(message)
        self.status = status
        self.outcome_unknown = outcome_unknown


# Use UTC timestamps as recorded data rather than deriving fake model timings.
def now():
    return datetime.now(timezone.utc).isoformat()


# Write state atomically with private file permissions.
def write_private(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# Read only bounded JSON files, rejecting duplicate fields and nonstandard numbers.
def read_bounded(path, limit=128 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise StoreError("Selected JSON file exceeds the size limit")
    raw = path.read_bytes()
    return decode(raw), hashlib.sha256(raw).hexdigest()


# Keep a verified outer generation failure visible without changing the recorded source projection.
def generation_status(record, receipt):
    if receipt is None:
        return
    if (not isinstance(receipt, dict) or receipt.get("status") not in {"ok", "failed", "interrupted"}
            or not isinstance(receipt.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])):
        raise StoreError("Invalid generation-stage display receipt")
    record["generation_stage_status"] = receipt["status"]
    record["generation_receipt_sha256"] = receipt["sha256"]
    if receipt["status"] != "ok":
        record["status"] = receipt["status"]
        record["warning"] = ("The saved generation stage " + receipt["status"] +
                             "; its recorded text is retained without an accepted completion or score.")


# Determine whether a recorded worker is still alive without restarting it.
def pid_alive(pid):
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# Persist display state separately from private workspaces and immutable benchmark runs.
class DemoStore:
    # Open a local operator store; live provider calls remain disabled unless the server opts in.
    def __init__(self, root, *, allow_live=False, model=DEFAULT_MODEL, top_k=8, disabled_reason=None):
        self.root = assert_operator_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.allow_live = bool(allow_live)
        self.disabled_reason = "" if self.allow_live else (disabled_reason or "Add API_KEY to .env and restart to enable new collaborations.")
        self.model = str(model)
        self.top_k = top_k
        self.closed = False
        self._thread_lock = threading.RLock()
        self._children = []
        with self._locked() as registry:
            if not registry["cases"]:
                projection = demo_projection()
                registry["cases"][projection["case_id"]] = self._demo_case(projection)

    # Serialize short registry transactions across the UI and detached worker processes.
    @contextmanager
    def _locked(self):
        with self._thread_lock:
            lock_path = self.root / "registry.lock"
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(fd, "r+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                path = self.root / "registry.json"
                registry = read_bounded(path)[0] if path.exists() else {"schema": "aos-demo-store-v1", "cases": {}, "runs": {}, "launches": {}}
                if registry.get("schema") != "aos-demo-store-v1":
                    raise StoreError("Unsupported conversation registry")
                yield registry
                write_private(path, registry)

    # Provide an explicitly authored zero-cost integration case, never a PiSAs score.
    def _demo_case(self, projection):
        return {"case_id": projection["case_id"], "title": "Offline availability demo",
                "task": projection["task"], "family": "authored_demo", "scenario": "availability-v1",
                "kind": "scripted", "participant_count": len(projection["participants"]),
                "source_sha256": hashlib.sha256(json.dumps(projection, sort_keys=True).encode()).hexdigest(),
                "scenario_path": None}

    # Register one scenario by reading only its scenario.json through the existing adapter.
    def add_case(self, scenario):
        path = Path(scenario).expanduser().resolve()
        path = path / "scenario.json" if path.is_dir() else path
        projection = load_scenario(path)
        record = {"case_id": projection["case_id"], "title": f"{projection['task_family']} / {projection['scenario_name']}",
                  "task": projection["task"], "family": projection["task_family"], "scenario": projection["scenario_name"],
                  "kind": "pisas", "participant_count": len(projection["participants"]),
                  "source_sha256": projection["source"]["scenario_sha256"], "scenario_path": str(path)}
        with self._locked() as registry:
            prior = registry["cases"].get(record["case_id"])
            if prior and prior["source_sha256"] != record["source_sha256"]:
                raise StoreError("Scenario contents changed; use a separate demo store for that revision")
            registry["cases"][record["case_id"]] = record
        return self._public_case(record)

    # Ensure the deterministic fixture remains available after catalogue imports.
    def add_demo(self):
        record = self._demo_case(demo_projection())
        with self._locked() as registry:
            registry["cases"][record["case_id"]] = record
        return self._public_case(record)

    # Project catalogue metadata without source paths or private artifacts.
    def _public_case(self, record):
        result = {key: record[key] for key in ("case_id", "title", "task", "family", "scenario", "kind", "participant_count", "source_sha256")}
        result["live_enabled"] = record["kind"] == "scripted" or self.allow_live
        return result

    # Return selectable cases and the server's explicit live-execution settings.
    def catalog(self):
        with self._locked() as registry:
            cases = [self._public_case(r) for r in registry["cases"].values()]
        return {"cases": sorted(cases, key=lambda c: (c["kind"] != "scripted", c["case_id"])),
                "versions": list(VERSIONS), "allow_live": self.allow_live, "model": self.model, "top_k": self.top_k,
                "live_disabled_reason": self.disabled_reason}

    # Expose only safe conversation metadata in the persistent sidebar.
    def _summary(self, record):
        keys = ("run_id", "title", "case_id", "version", "model", "run_index", "run_index_origin", "mode", "status", "created", "updated", "evaluation_status", "backend", "generation_stage_status", "generation_receipt_sha256", "generation_error_type", "cohort", "date_origin")
        result = {key: record.get(key) for key in keys}
        if (record.get("response_contract_id") == RESPONSE_CONTRACT_ID
                and record.get("version") in {"governed-v1", "integrated-v3"}):
            result["response_contract_id"] = RESPONSE_CONTRACT_ID
        return result

    # Mark a worker that disappeared as interrupted while preserving every recorded result.
    def _refresh_record(self, record):
        if record["mode"] == "live" and record["status"] in ACTIVE and record.get("worker_pid") and not pid_alive(record["worker_pid"]):
            record.update(status="interrupted", updated=now())
            record["warning"] = "The execution worker exited without a terminal receipt. Inspect the saved trace; no automatic restart."
        return record

    # List all attempted runs, including failures and interrupted work, without loading private traces.
    def runs(self):
        for child in self._children:
            child.poll()
        with self._locked() as registry:
            records = [self._refresh_record(r) for r in registry["runs"].values()]
            result = [self._summary(r) for r in records]
            active = next((r["run_id"] for r in records if r["status"] in ACTIVE), None)
        return {"runs": sorted(result, key=lambda r: r["created"], reverse=True), "active_run_id": active}

    # Resolve only registry-owned identifiers to internal directories.
    def _directory(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise StoreError("Invalid conversation ID", 404)
        return self.root / "runs" / run_id

    # Import a historical run as an immutable requester projection without copying its private source corpus.
    def add_replay(self, source, *, title=None, pipeline=None, evaluation=None, generation_receipt=None):
        run, digest = read_bounded(Path(source).expanduser().resolve())
        run_id = "run-" + uuid.uuid4().hex
        view = build_view(run, run_id, digest)
        config = run["config"]
        version = "observed-v2" if config["protocol"] == "aos-artifact-requests-observed-v2" else "governed-v1"
        if config.get("system_version") == "integrated-v3":
            version = "integrated-v3"
        if config.get("mode") == "prompt_only":
            version = "prompt-only-v1"
        record = {"run_id": run_id, "title": title or view["scenario"]["case_id"], "case_id": view["scenario"]["case_id"],
                  "version": version, "model": config.get("model", "unknown"), "run_index": config.get("run_index"),
                  "mode": "recorded_replay", "status": run["status"], "created": now(), "updated": now(),
                  "evaluation_status": "not_loaded", "source_sha256": digest, "backend": config.get("backend", "unknown")}
        if "response_contract_id" in view:
            record["response_contract_id"] = view["response_contract_id"]
        generation_status(record, generation_receipt)
        # Historical judgments are attached outside the actor's view and are never execution inputs.
        attached = {"status": "not_loaded"}
        if evaluation is not None:
            if generation_receipt is not None and generation_receipt["status"] != "ok":
                raise StoreError("An unsuccessful generation stage cannot acquire a completed evaluation")
            attached = attach_evaluation(evaluation, pipeline, view, record)
            record["evaluation_status"] = "attached"
        elif pipeline is not None:
            raise StoreError("A pipeline attachment requires its saved judgment")
        directory = self._directory(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        os.chmod(directory, 0o700)
        write_private(directory / "view.json", view)
        write_private(directory / "evaluation.json", attached)
        with self._locked() as registry:
            registry["runs"][run_id] = record
        return self._summary(record)

    # Import native peer messages through their own faithful projection and evaluation binding.
    def add_native_replay(self, source, *, case, run_index=0, title=None, evaluation=None, generation_receipt=None):
        pipeline_path = Path(source).expanduser().resolve()
        pipeline, digest = read_bounded(pipeline_path)
        run_id = "run-" + uuid.uuid4().hex
        view = build_native_view(pipeline, run_id, digest, case=case)
        config = pipeline["config"]
        if type(run_index) is not int or run_index < 0:
            raise StoreError("Native run index must be a nonnegative integer")
        if config.get("run_index", run_index) != run_index:
            raise StoreError("Native pipeline run index differs from the selected replay")
        model = config.get("agent_llm")
        if not isinstance(model, str) or not model.strip():
            raise StoreError("Native pipeline lacks its actor model")
        state = view["snapshot_status"]
        if state == "completed" and view["degraded"]:
            state = "completed_with_warning"
        record = {"run_id": run_id, "title": title or case, "case_id": view["scenario"]["case_id"],
                  "version": "native-decentralized", "model": model, "run_index": run_index,
                  "run_index_origin": "pipeline_config" if "run_index" in config else "operator_selected",
                  "mode": "native_replay", "status": state, "created": now(), "updated": now(),
                  "evaluation_status": "not_loaded", "source_sha256": digest, "backend": "native"}
        if view["degraded"]:
            record["warning"] = "The native pipeline recorded degraded execution; replay preserves the saved result."
        generation_status(record, generation_receipt)
        attached = {"status": "not_loaded"}
        if evaluation is not None:
            if view["snapshot_status"] != "completed" or (generation_receipt is not None and generation_receipt["status"] != "ok"):
                raise StoreError("An incomplete native replay cannot acquire a completed evaluation")
            attached = attach_native_evaluation(evaluation, pipeline_path, view, record, case=case)
            record["evaluation_status"] = "attached"
        directory = self._directory(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        os.chmod(directory, 0o700)
        write_private(directory / "view.json", view)
        write_private(directory / "evaluation.json", attached)
        with self._locked() as registry:
            registry["runs"][run_id] = record
        return self._summary(record)

    # Reuse an exact historical import bundle without creating duplicate sidebar conversations.
    def find_replay(self, source_sha256, *, pipeline_sha256=None, judgment_sha256=None, generation_receipt_sha256=None):
        with self._locked() as registry:
            for record in registry["runs"].values():
                if record["mode"] not in {"recorded_replay", "native_replay"} or record.get("source_sha256") != source_sha256:
                    continue
                if record.get("generation_receipt_sha256") != generation_receipt_sha256:
                    continue
                attached_path = self._directory(record["run_id"]) / "evaluation.json"
                attached = read_bounded(attached_path)[0] if attached_path.exists() else {}
                if (attached.get("pipeline_sha256") == pipeline_sha256
                        and attached.get("judgment_sha256") == judgment_sha256):
                    return self._summary(record)
        return None

    # Return the last valid requester projection while an atomic runtime snapshot is being updated.
    def snapshot(self, run_id):
        for child in self._children:
            child.poll()
        directory = self._directory(run_id)
        with self._locked() as registry:
            if run_id not in registry["runs"]:
                raise StoreError("Unknown conversation", 404)
            record = copy.deepcopy(self._refresh_record(registry["runs"][run_id]))
        view_path = directory / "view.json"
        view = read_bounded(view_path)[0] if view_path.exists() else None
        warnings = [record["warning"]] if record.get("warning") else []
        if record["mode"] == "live":
            raw_path = directory / "actor" / "run.json"
            if raw_path.exists():
                try:
                    run, digest = read_bounded(raw_path)
                    candidate = build_view(run, run_id, digest)
                    candidate["mode"] = "live"
                    write_private(view_path, candidate)
                    view = candidate
                except (OSError, ValueError, TypeError, KeyError):
                    warnings.append("The latest execution snapshot is incomplete or invalid; displaying the last validated view.")
        evaluation_path = directory / "evaluation.json"
        evaluation = read_bounded(evaluation_path)[0] if evaluation_path.exists() else {"status": "not_loaded"}
        progress_path = directory / "progress.json"
        progress = read_bounded(progress_path)[0] if progress_path.exists() else {"phase": record["status"], "message": "Waiting for an execution record", "updated": record["updated"]}
        return {"metadata": self._summary(record), "view": view, "progress": progress,
                "evaluation": evaluation, "warnings": warnings}

    # Launch one detached worker so browser closure does not interrupt or duplicate the conversation.
    def _launch(self, run_id, directory):
        stream = (directory / "worker.log").open("ab")
        os.chmod(directory / "worker.log", 0o600)
        try:
            process = subprocess.Popen([sys.executable, "-B", "-m", "aos.collaboration.demo", "execute", str(self.root), run_id],
                                       cwd=Path(__file__).resolve().parents[3], stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            self._children.append(process)
            return process.pid
        finally:
            stream.close()

    # Persist an idempotent start receipt before allowing any worker to make provider calls.
    def start(self, case_id, version, launch_id):
        if self.closed:
            raise StoreError("This UI is shutting down", 503)
        if version not in VERSIONS:
            raise StoreError("Unknown system version")
        if not isinstance(launch_id, str) or not LAUNCH_ID.fullmatch(launch_id):
            raise StoreError("A valid launch ID is required")
        with self._locked() as registry:
            prior_id = registry["launches"].get(launch_id)
            if prior_id:
                prior = registry["runs"][prior_id]
                if prior["case_id"] != case_id or prior["version"] != version:
                    raise StoreError("Launch ID was already used for another request", 409)
                return {"run_id": prior_id, "status": prior["status"], "duplicate": True}
            case = registry["cases"].get(case_id)
            if not case:
                raise StoreError("Unknown case", 404)
            if case["kind"] == "scripted" and version == "integrated-v3":
                raise StoreError("Integrated v3 requires a PiSAs scenario")
            if case["kind"] == "pisas" and not self.allow_live:
                raise StoreError("Live model execution is disabled. " + self.disabled_reason, 403)
            if any(self._refresh_record(r)["status"] in ACTIVE for r in registry["runs"].values()):
                raise StoreError("A conversation is already running; wait for its terminal record", 409)
            run_id = "run-" + uuid.uuid4().hex
            directory = self._directory(run_id)
            directory.mkdir(parents=True, exist_ok=False)
            os.chmod(directory, 0o700)
            record = {"run_id": run_id, "title": case["title"], "case_id": case_id,
                      "version": version, "model": "scripted" if case["kind"] == "scripted" else self.model,
                      "run_index": 0, "mode": "live", "status": "queued", "created": now(), "updated": now(),
                      "evaluation_status": "not_loaded", "case": copy.deepcopy(case),
                      "top_k": None if version == "integrated-v3" else self.top_k,
                      "backend": "scripted" if case["kind"] == "scripted" else "model"}
            if version == "integrated-v3":
                record["response_contract_id"] = RESPONSE_CONTRACT_ID
            registry["runs"][run_id] = record
            registry["launches"][launch_id] = run_id
            # The worker waits for this registry transaction before reading its frozen launch settings.
            try:
                record["worker_pid"] = self._launch(run_id, directory)
            except OSError:
                record.update(status="failed", updated=now(), warning="The worker could not be launched; no actor was started.")
                return {"run_id": run_id, "status": "failed", "duplicate": False}
            return {"run_id": run_id, "status": "queued", "duplicate": False}

    # Stop accepting new UI launches (already accepted detached work keeps its recorded identity).
    def close(self):
        self.closed = True
        for process in self._children:
            process.poll()

    # Execute the exact queued request once in its background worker and persist its terminal state.
    def execute(self, run_id):
        directory = self._directory(run_id)
        with self._locked() as registry:
            record = registry["runs"].get(run_id)
            if not record or record["status"] != "queued":
                raise StoreError("This worker request is not eligible for execution")
            if record.get("worker_pid") != os.getpid():
                raise StoreError("This process does not own the accepted request")
            record.update(status="running", updated=now())
            frozen = copy.deepcopy(record)
        # A fixed public progress message accompanies each actual coordinator callback.
        def progress(message):
            text = str(message)
            write_private(directory / "progress.json", {"phase": "running", "message": text[:600], "updated": now()})
            print(text, flush=True)
        try:
            case = frozen["case"]
            if case["scenario_path"] is not None:
                path = Path(case["scenario_path"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != case["source_sha256"]:
                    raise StoreError("The registered scenario changed before execution")
            else:
                path = None
            from .runner import execute_case
            result = execute_case(path, directory / "actor", version=frozen["version"],
                                  model=frozen["model"] if path is not None else DEFAULT_MODEL,
                                  top_k=frozen["top_k"], progress=progress)
            state = result.get("status", "failed")
            if state not in {"completed", "failed"}:
                raise StoreError("Execution returned no terminal state")
            run, digest = read_bounded(directory / "actor" / "run.json")
            view = build_view(run, run_id, digest)
            view["mode"] = "live"
            write_private(directory / "view.json", view)
            note = "Conversation completed" if state == "completed" else "Conversation failed; inspect its saved trace"
        except Exception as exc:
            state = "failed"
            note = f"Execution failed ({type(exc).__name__}); inspect the private worker log"
            print(note, flush=True)
            traceback.print_exc()
            # Preserve any valid partial run while never publishing a private failed decision draft.
            try:
                run, digest = read_bounded(directory / "actor" / "run.json")
                view = build_view(run, run_id, digest)
                view["mode"] = "live"
                write_private(directory / "view.json", view)
            except (OSError, ValueError, TypeError, KeyError):
                pass
        with self._locked() as registry:
            registry["runs"][run_id].update(status=state, updated=now())
        write_private(directory / "progress.json", {"phase": state, "message": note, "updated": now()})
        return {"run_id": run_id, "status": state}
