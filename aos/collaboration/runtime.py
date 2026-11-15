# Run one fresh artifact-request experiment and retain honest execution evidence.
# The coordinator asks every owner the same task-derived question. It never sends
# one owner's released facts onward to a different owner. This is an explicitly
# named system intervention, not PiSAs' original token-passing implementation.

from __future__ import annotations
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from ..llm import add_usage
from ..tasks import Step, Task
from ..workspace import Workspace
from .common import canonical, read_json, timestamp, write_json
from .experiment import prepare_experiment
from .fixture import FixtureModel
from .model import call_json, create_model, usage_delta, usage_snapshot, visible_context


# Preserve a failed run on disk while preventing its export as a successful benchmark trace.
class RunFailed(RuntimeError):
    pass


# Prevent a configured endpoint from redirecting credentials to another service.
class _NoRedirect(HTTPRedirectHandler):
    # Refuse redirects rather than forwarding a bearer credential to a new origin.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# Require a loopback URL with no embedded credentials or unexpected path prefix.
def _endpoint(value: str) -> str:
    url = urlparse(value)
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
            or url.username or url.password or url.path not in {"", "/"} or url.query or url.fragment):
        raise ValueError("Expected an explicit loopback service endpoint")
    return value.rstrip("/")


# Make one request with no proxy, no redirect, and no automatic delivery retry.
def request_json(url: str, token: str, data: dict | None = None, *, timeout: float = 210) -> dict:
    request = Request(url, data=None if data is None else canonical(data).encode("utf-8"),
                      headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
                      method="GET" if data is None else "POST")
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
    except HTTPError as exc:
        raise RunFailed(f"Artifact service returned HTTP {exc.code}; inspect its private audit") from None
    if len(raw) > 8 * 1024 * 1024:
        raise RunFailed("Service response exceeds the eight-megabyte limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RunFailed("Service response is not a JSON object")
    return value


# Validate the narrow response schema before exposing it to the requesting actor.
def validate_response(value: dict, request_id: str) -> dict:
    if (set(value) != {"request_id", "status", "answer", "claims", "coverage"}
            or value["request_id"] != request_id or value["coverage"] != "not_established"
            or not isinstance(value["claims"], list) or len(value["claims"]) > 32):
        raise RunFailed("Service returned an invalid release response")
    texts = []
    for claim in value["claims"]:
        if (not isinstance(claim, dict) or set(claim) != {"text", "evidence_refs"}
                or not isinstance(claim["text"], str) or not claim["text"].strip()
                or len(claim["text"]) > 2000 or not isinstance(claim["evidence_refs"], list)
                or not claim["evidence_refs"]
                or any(not isinstance(ref, str) or not ref.startswith("ref-") or len(ref) != 28
                       for ref in claim["evidence_refs"])):
            raise RunFailed("Service returned an invalid released claim")
        texts.append(claim["text"])
    if value["answer"] != "\n".join(texts) or value["status"] != ("released" if texts else "unavailable"):
        raise RunFailed("Service response contains unreviewed text or inconsistent status")
    return value


# Launch distinct owner processes or connect to already running authenticated services.
class ServiceGroup:
    # Keep process handles outside all actor-visible state.
    def __init__(self, root: Path, manifest: dict, *, external: bool = False):
        self.root, self.manifest, self.external = root, manifest, external
        self.processes = []
        self.streams = []
        self.urls = {}

    # Wait for validated readiness without starting any model conversations.
    def start(self) -> dict[str, str]:
        try:
            for owner in self.manifest["owners"]:
                if not self.external:
                    log = (self.root / owner["config"]).parent / "service.log"
                    stream = log.open("ab")
                    self.streams.append(stream)
                    process = subprocess.Popen(
                        [sys.executable, "-m", "aos.collaboration", "serve", str(self.root / owner["config"])],
                        stdout=stream, stderr=stream, cwd=str(Path(__file__).resolve().parents[2]))
                    self.processes.append(process)
                deadline = time.monotonic() + 30
                last_error = None
                while time.monotonic() < deadline:
                    if not self.external and self.processes[-1].poll() is not None:
                        raise RunFailed(f"Service {owner['id']} exited during startup; inspect its service.log")
                    try:
                        endpoint = read_json(self.root / owner["endpoint"])
                        if endpoint["owner_id"] != owner["id"] or endpoint["case_id"] != self.manifest["case_id"]:
                            raise RunFailed("Endpoint identity does not match this experiment")
                        if not self.external and endpoint["pid"] != self.processes[-1].pid:
                            raise RunFailed("Endpoint receipt belongs to an earlier service process")
                        url = _endpoint(endpoint["url"])
                        health = request_json(url + "/health", owner["request_token"], timeout=1)
                        if health != {"ready": True, "owner_id": owner["id"], "case_id": self.manifest["case_id"]}:
                            raise RunFailed("Invalid service health receipt")
                        self.urls[owner["id"]] = url
                        break
                    except (OSError, ValueError, KeyError, RunFailed) as exc:
                        last_error = type(exc).__name__
                        time.sleep(0.1)
                else:
                    raise RunFailed(f"Service {owner['id']} did not become ready ({last_error})")
            return self.urls
        except Exception:
            self.stop()
            raise

    # Terminate only subprocesses created by this coordinator, never external services.
    def stop(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for stream in self.streams:
            stream.close()


# Validate the final decision shape while leaving semantic correctness to evaluation.
def _decision(value: dict, refs: set[str]) -> dict:
    if set(value) != {"final_response", "used_evidence_refs", "insufficient_evidence"}:
        raise RunFailed("Decision model returned an invalid response schema")
    if (not isinstance(value["final_response"], str) or not value["final_response"].strip()
            or len(value["final_response"]) > 16000 or type(value["insufficient_evidence"]) is not bool
            or not isinstance(value["used_evidence_refs"], list)
            or any(not isinstance(ref, str) or ref not in refs for ref in value["used_evidence_refs"])):
        raise RunFailed("Decision model returned invalid text or unknown evidence references")
    return value


# Store the executor's released evidence and decision as a normal native AOS task.
def _record_decision(root: Path, manifest: dict, result: dict) -> None:
    owner = next(item for item in manifest["owners"] if item["id"] == manifest["executor_id"])
    ws = Workspace.open(root / owner["workspace"])
    # Owner services have already stopped, so this is the sole writer for this step.
    lock_path = ws.root.parent / f".{ws.root.name}.artifact.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        task = Task.new(ws, "Decide from released artifact evidence")
        record = task.start_step(Step(worker="artifact_decision", instructions=result["projection"]["task"],
                                     reason="Use a fresh context containing released evidence only"))
        task.log_path(1).write_text("Decision used released claims only; private model-call evidence is in the operator audit.\n", encoding="utf-8")
        payloads = {"generated/collaboration/gathered.txt": result["gathered_info"],
                    "generated/collaboration/answer.txt": result["final_response"]}
        for rel, text in payloads.items():
            ws.abspath(rel).parent.mkdir(parents=True, exist_ok=True)
            ws.abspath(rel).write_text(text + "\n", encoding="utf-8")
            ws.register(rel, source="artifact_decision", task=task.id, step=1,
                        reason="Decision from released evidence", summary="Artifact collaboration result")
        task.finish_step(record, status="ok", outputs=list(payloads), summary=result["final_response"],
                         usage=result.get("decision_usage"))
        task.finish("done", result["final_response"], usage=result.get("decision_usage"))
        record["commit"] = ws.checkpoint(f"{task.id}: artifact decision") or ws.head()
        task.save()
        result["decision_task_id"] = task.id


# Run one fresh scenario, persisting failures and every actual message before evaluation.
def run_experiment(root: str | Path, *, external_services: bool = False, model=None,
                   progress=None) -> dict:
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    projection = read_json(root / manifest["projection"])
    run_path = root / "run.json"
    if run_path.exists():
        raise RunFailed("This experiment already has a run; prepare a fresh directory for another run")
    guard = (root / ".run.lock").open("a+")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        guard.close()
        raise RunFailed("Another coordinator owns this experiment") from None
    if run_path.exists():
        guard.close()
        raise RunFailed("This experiment already has a run")
    start = time.monotonic()
    result = {"status": "running", "projection": projection, "config": manifest["config"],
              "events": [], "agent_contents": {p["id"]: "" for p in projection["participants"]},
              "responses": [], "final_response": "", "gathered_info": "", "n_a2a": 0,
              "started": timestamp(), "degraded": False, "owner_usage": {}, "decision_audit": []}
    write_json(run_path, result)
    group = ServiceGroup(root, manifest, external=external_services)
    executor = manifest["executor_id"]
    try:
        urls = group.start()
        decision_model = model or (FixtureModel() if manifest["config"]["backend"] == "scripted"
                                   else create_model(manifest["config"]["model"]))
        for owner in manifest["owners"]:
            request_id = "req-" + uuid.uuid4().hex
            request = {"request_id": request_id, "case_id": manifest["case_id"],
                       "question": projection["task"]}
            request_event = None
            if owner["id"] != executor:
                request_event = {"sender_id": executor, "recipient_id": owner["id"],
                    "request_id": request_id, "kind": "request", "body": request["question"],
                    "delivery": "attempted", "timestamp": timestamp()}
                result["events"].append(request_event)
                result["n_a2a"] += 1
            write_json(run_path, result)
            if progress:
                progress(f"Requesting artifacts from {owner['name']} ({owner['id']})")
            try:
                response = validate_response(request_json(urls[owner["id"]] + "/request",
                                             owner["request_token"], request), request_id)
                if request_event is not None:
                    request_event["delivery"] = "confirmed"
            finally:
                # This privileged read feeds evaluation only, never another owner's model.
                try:
                    audit = request_json(urls[owner["id"]] + "/audit?" + urlencode({"request_id": request_id}),
                                         owner["admin_token"], timeout=5)
                    result["agent_contents"][owner["id"]] += audit.get("agent_content", "")
                    result["owner_usage"][owner["id"]] = audit.get("usage", {})
                    if request_event is not None and audit.get("request_id") == request_id:
                        request_event["delivery"] = "confirmed"
                except Exception:
                    result["degraded"] = True
            result["responses"].append({"owner_id": owner["id"], "response": response})
            if owner["id"] != executor:
                body = canonical(response)
                result["events"].append({"sender_id": owner["id"], "recipient_id": executor,
                    "request_id": request_id, "kind": "response", "body": body,
                    "delivery": "confirmed", "timestamp": timestamp()})
                result["n_a2a"] += 1
                result["agent_contents"][executor] += "\nReceived artifact response:\n" + body
            write_json(run_path, result)
            if progress:
                progress(f"{owner['name']}: {response['status']} ({len(response['claims'])} released claims)")
        if result["degraded"]:
            raise RunFailed("One or more private audit channels could not be collected")
        claims = [claim for item in result["responses"] for claim in item["response"]["claims"]]
        result["gathered_info"] = "\n".join(
            f"{claim['text']} [evidence: {', '.join(claim['evidence_refs'])}]" for claim in claims)
        refs = sorted({ref for claim in claims for ref in claim["evidence_refs"]})
        before = usage_snapshot(decision_model)
        try:
            decision = _decision(call_json(decision_model, "decide", {
                "task": projection["task"], "recipient": projection["recipient"],
                "participants": projection["participants"], "gathered_info": result["gathered_info"],
                "evidence_refs": refs,
            }, result["decision_audit"]), set(refs))
        finally:
            result["decision_usage"] = usage_delta(decision_model, before)
            result["agent_contents"][executor] += "\n" + visible_context(result["decision_audit"])
        result.update(decision)
        result["usage"] = add_usage(*result["owner_usage"].values(), result["decision_usage"])
        group.stop()
        if not external_services:
            _record_decision(root, manifest, result)
        # External services still own their workspaces; do not violate their writer locks.
        result.update(status="completed", finished=timestamp(), pipeline_time=time.monotonic() - start)
        write_json(run_path, result)
        return result
    except BaseException as exc:
        group.stop()
        result.update(status="failed", error=type(exc).__name__, finished=timestamp(),
                      pipeline_time=time.monotonic() - start)
        result["usage"] = add_usage(*result["owner_usage"].values(), result.get("decision_usage"))
        write_json(run_path, result)
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RunFailed(f"Run failed ({type(exc).__name__}); inspect {run_path}") from exc
    finally:
        guard.close()


# Summarize a saved run without exposing private contexts or calling a model.
def inspect_experiment(root: str | Path) -> dict:
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if not (root / "run.json").exists():
        return {"case_id": manifest["case_id"], "status": "prepared", "owners": len(manifest["owners"])}
    run = read_json(root / "run.json")
    return {"case_id": manifest["case_id"], "status": run["status"], "config": run["config"],
            "n_a2a": run["n_a2a"], "releases": [{"owner_id": item["owner_id"],
                "status": item["response"]["status"], "claims": len(item["response"]["claims"])}
                for item in run["responses"]], "final_response": run["final_response"],
            "insufficient_evidence": run.get("insufficient_evidence"), "usage": run.get("usage"),
            "error": run.get("error"), "degraded": run.get("degraded", False),
            "benchmark_score": "not_evaluated"}


# Verify the authored offline fixture without treating it as measured model performance.
def verify_demo(root: str | Path) -> dict:
    run = read_json(Path(root) / "run.json")
    if run["projection"].get("fixture") != "availability-v1" or run["status"] != "completed":
        raise RunFailed("Expected a completed authored availability demo")
    public = canonical({"events": run["events"], "responses": run["responses"],
                        "final_response": run["final_response"], "decision_audit": run["decision_audit"]})
    if "DEMO_PRIVATE" in public:
        raise RunFailed("The private fixture marker crossed the release boundary")
    if (run["n_a2a"] != 2 or len(run["responses"]) != 2
            or any(len(item["response"]["claims"]) != 1 for item in run["responses"])
            or "10:00" not in run["final_response"] or "10:30" not in run["final_response"]):
        raise RunFailed("The expected useful scheduling exchange did not complete")
    return {"mechanism_verified": True, "benchmark_score": "not_evaluated",
            "backend": run["config"]["backend"], "n_a2a": 2,
            "final_response": run["final_response"]}
