# Authenticated loopback artifact requests with one serial writer per owner.
# This service deliberately exposes no raw-file, shell, metadata-SQL, or arbitrary
# send-message endpoint. The operator and configured model provider are trusted.

from __future__ import annotations
import fcntl
import hmac
import json
import os
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from ..tasks import Step, Task
from ..workspace import Workspace
from .artifacts import ArtifactIndex, EvidenceBundle
from .common import canonical, digest, identifier, outside_workspace, read_json, timestamp, write_json
from .fixture import FixtureModel
from .model import call_json, create_model, usage_delta, usage_snapshot, visible_context
from .policy import PolicyStore, RequestContext
from .release import ReleaseProposal, ReviewRecord, release

MAX_BODY = 65_536
STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, requester TEXT NOT NULL, fingerprint TEXT NOT NULL,
    status TEXT NOT NULL, response TEXT, task_id TEXT, created TEXT NOT NULL
);
"""


# Distinguish malformed input, uncertain execution, and duplicate-request conflicts.
class RequestFailure(RuntimeError):
    # Carry only a fixed public error code across the HTTP boundary.
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code
        super().__init__(code)


# Own one AOS workspace and answer narrow, identity-bound artifact requests.
class ArtifactService:
    # Validate private configuration before acquiring the workspace and loading a model.
    def __init__(self, config_path: str | Path, model=None):
        self.config_path = Path(config_path).resolve()
        self.directory = self.config_path.parent
        self.config = read_json(self.config_path)
        config = self.config
        if config.get("schema_version") != 1:
            raise ValueError("Unsupported service configuration")
        self.owner_id = identifier(config["owner_id"])
        identifier(config["case_id"])
        if config.get("mode") not in {"governed", "prompt_only"}:
            raise ValueError("Unknown service mode")
        self.workspace = (self.directory / config["workspace"]).resolve()
        self.ws = Workspace.open(self.workspace)
        self.paths = {key: outside_workspace(self.directory / config[key], self.workspace)
                      for key in ("index", "policy", "state", "endpoint")}
        self.credentials = config["credentials"]
        if not isinstance(self.credentials, dict) or not self.credentials:
            raise ValueError("At least one bound requester credential is required")
        tokens = [*self.credentials.values(), config["admin_token"]]
        if any(not isinstance(token, str) or len(token) < 24 for token in tokens) or len(set(tokens)) != len(tokens):
            raise ValueError("Service credentials must be distinct random secrets of at least 24 characters")
        for principal in self.credentials:
            identifier(principal)
        if not isinstance(config["task"], str) or not config["task"].strip():
            raise ValueError("A configured task/purpose is required")
        top_k = config.get("top_k", 8)
        if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 500):
            raise ValueError("top_k must be 1..500 or null")
        self._lock = (self.workspace.parent / f".{self.workspace.name}.artifact.lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise RuntimeError("Another artifact service owns this workspace") from None
        self._thread = None
        self._busy = threading.Lock()
        self._stopping = threading.Event()
        try:
            self.index = ArtifactIndex(self.paths["index"], self.workspace, self.owner_id)
            for artifact in config["artifacts"]:
                self.index.register_file(artifact["path"], artifact["artifact_id"],
                                         kind=artifact.get("type", "text"), source="scenario")
            self.index.retain_artifacts([item["artifact_id"] for item in config["artifacts"]])
            self.policy = PolicyStore(self.paths["policy"], workspace=self.workspace, owner_id=self.owner_id)
            policy = self.policy.load()
            if policy.mode == "explicit_fixture" and config.get("fixture") != "availability-v1":
                raise ValueError("Explicit fixture policy is only enabled for the authored offline demo")
            if config.get("backend") == "scripted" and config.get("fixture") != "availability-v1":
                raise ValueError("The scripted backend cannot evaluate PiSAs")
            self.model = model or (FixtureModel() if config.get("backend") == "scripted"
                                   else create_model(config["model"], config.get("request_timeout", 90)))
            with sqlite3.connect(self.paths["state"]) as connection:
                connection.executescript(STATE_SCHEMA)
                connection.execute("UPDATE requests SET status='interrupted' WHERE status='processing'")
            Task.mark_interrupted(self.ws)
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            self.server.daemon_threads = True
            self.server.service = self
        except Exception:
            self._lock.close()
            raise

    # Report the bound loopback address without relying on a guessed free port.
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    # Start HTTP processing and publish a private endpoint receipt for the coordinator.
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Service already started")
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        write_json(self.paths["endpoint"], {"url": self.url, "owner_id": self.owner_id,
                                          "case_id": self.config["case_id"], "pid": os.getpid()})

    # Stop accepting requests and release ownership after active processing has finished.
    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self.server.shutdown()
            self._thread.join(timeout=2)
        self.server.server_close()
        if not self._busy.locked():
            self._lock.close()

    # Resolve a credential to a principal rather than trusting a sender in request JSON.
    def authenticate(self, token: str, *, admin: bool = False) -> str | None:
        if admin:
            return "operator" if hmac.compare_digest(token, self.config["admin_token"]) else None
        for principal, expected in self.credentials.items():
            if hmac.compare_digest(token, expected):
                return principal
        return None

    # Read a request's private audit only through the operator-authenticated route.
    def audit(self, request_id: str) -> dict:
        identifier(request_id)
        path = self.directory / "audit" / f"{request_id}.json"
        if not path.is_file():
            raise RequestFailure(404, "audit_unavailable")
        return read_json(path)

    # Persist canonical response status without overwriting the original request binding.
    def _finish(self, request_id: str, status: str, response: dict | None, task_id: str | None) -> None:
        with sqlite3.connect(self.paths["state"]) as connection:
            connection.execute("UPDATE requests SET status=?,response=?,task_id=? WHERE request_id=?",
                               (status, canonical(response) if response is not None else None, task_id, request_id))

    # Accept a new request once or return its exact previously committed response.
    def handle_request(self, principal: str, data: dict) -> dict:
        if (not isinstance(data, dict) or set(data) != {"request_id", "case_id", "question"}
                or data["case_id"] != self.config["case_id"]
                or not isinstance(data["question"], str) or not data["question"].strip()
                or len(data["question"]) > 16_000 or principal not in self.credentials):
            raise RequestFailure(400, "invalid_request")
        try:
            request_id = identifier(data["request_id"])
        except ValueError:
            raise RequestFailure(400, "invalid_request") from None
        authorization_context = {key: self.config.get(key) for key in
                                 ("task", "participants", "relations", "mode", "case_id", "artifacts")}
        fingerprint = digest({"requester": principal, "request": data,
                              "authorization_context": authorization_context})
        if not self._busy.acquire(blocking=False):
            raise RequestFailure(409, "owner_busy")
        try:
            if self._stopping.is_set():
                raise RequestFailure(503, "owner_stopping")
            with sqlite3.connect(self.paths["state"]) as connection:
                row = connection.execute("SELECT requester,fingerprint,status,response FROM requests WHERE request_id=?",
                                         (request_id,)).fetchone()
                if row is not None:
                    if row[0] != principal or row[1] != fingerprint:
                        raise RequestFailure(409, "request_id_conflict")
                    if row[2] == "completed":
                        saved = self.audit(request_id)
                        if (saved["context"]["purpose"] != self.config["task"]
                                or saved["proposal"]["policy_version"] != self.policy.load().version):
                            raise RequestFailure(409, "stale_response")
                        try:
                            for unit in saved["retrieved"]["units"]:
                                if self.index.read(unit["evidence_id"]).as_dict() != unit:
                                    raise ValueError("Changed source")
                        except ValueError:
                            raise RequestFailure(409, "stale_response") from None
                        return json.loads(row[3])
                    raise RequestFailure(409, "request_not_replayable")
                connection.execute("INSERT INTO requests VALUES (?,?,?,'processing',NULL,NULL,?)",
                                   (request_id, principal, fingerprint, timestamp()))
            try:
                return self._process(principal, data)
            except RequestFailure:
                raise
            except Exception:
                self._finish(request_id, "failed", None, None)
                raise RequestFailure(500, "request_failed") from None
        finally:
            self._busy.release()
            if self._stopping.is_set():
                self._lock.close()

    # Retrieve, propose, review, and commit a canonical response with native AOS task records.
    def _process(self, principal: str, data: dict) -> dict:
        context = RequestContext(data["request_id"], principal, self.owner_id,
                                 self.config["task"], data["question"])
        audit = {"request_id": context.request_id, "context": context.as_dict(),
                 "status": "processing", "events": [], "started": timestamp()}
        task = Task.new(self.ws, "Answer an authenticated artifact request")
        record = task.start_step(Step(worker="artifact_request", instructions=data["question"],
                                     reason="Retrieve evidence and check disclosure before release"))
        before = usage_snapshot(self.model)
        task.log_path(1).write_text(f"Artifact request {context.request_id}\nRequester: {principal}\n", encoding="utf-8")
        try:
            units = self.index.search(data["question"], limit=self.config.get("top_k", 8))
            bundle = EvidenceBundle.create(context, units)
            policy = self.policy.load()
            if policy.mode == "explicit_fixture" and self.config.get("fixture") != "availability-v1":
                raise ValueError("Unexpected fixture policy")
            record["inputs"] = sorted({unit.source_path for unit in units})
            task.save()
            audit["retrieved"] = bundle.as_dict()
            extracted = call_json(self.model, "extract", {
                "context": context.as_dict(), "participants": self.config["participants"],
                "relations": self.config.get("relations", []), "policy_instructions": policy.instructions,
                "evidence": [unit.as_dict() for unit in units],
            }, audit["events"]) if units else {"claims": []}
            if set(extracted) != {"claims"}:
                raise ValueError("Extraction must return claims only")
            proposal = ReleaseProposal.create(context, bundle, policy, extracted["claims"])
            audit["proposal"] = proposal.as_dict()
            if proposal.claims and self.config["mode"] == "governed":
                reviewed = call_json(self.model, "review", {
                    "context": context.as_dict(), "participants": self.config["participants"],
                    "relations": self.config.get("relations", []), "policy_instructions": policy.instructions,
                    "proposal": proposal.as_dict(),
                }, audit["events"])
            else:
                # This named ablation intentionally omits the independent semantic review.
                reviewed = {"decisions": [{"claim_id": claim.claim_id, "supported": True,
                                            "appropriate": True, "permitted": True}
                                           for claim in proposal.claims]}
            review = ReviewRecord.from_dict(proposal, reviewed)
            audit["review"] = review.as_dict()
            response = release(self.index, self.policy, context, proposal, review).as_dict()
            relative = f"generated/responses/{context.request_id}.json"
            write_json(self.ws.abspath(relative), response)
            self.ws.register(relative, source="artifact_request", agent=self.owner_id, task=task.id,
                             step=1, reason="Canonical checked artifact response",
                             summary="Released claims and opaque evidence references", inputs=record["inputs"])
            usage = usage_delta(self.model, before)
            task.finish_step(record, status="ok", outputs=[relative],
                             summary=f"Artifact response: {response['status']}", usage=usage)
            task.finish("done", response["answer"] or "No releasable evidence", usage=usage)
            record["commit"] = self.ws.checkpoint(f"{task.id}: artifact response committed") or self.ws.head()
            task.save()
            audit.update(status="completed", response=response, task_id=task.id, usage=usage,
                         agent_content=visible_context(audit["events"]), finished=timestamp())
            write_json(self.directory / "audit" / f"{context.request_id}.json", audit)
            self._finish(context.request_id, "completed", response, task.id)
            return response
        except Exception as exc:
            usage = usage_delta(self.model, before)
            code = type(exc).__name__
            task.finish_step(record, status="failed", outputs=[], summary="", error=code, usage=usage)
            task.finish("failed", f"Artifact request failed ({code})", usage=usage)
            self.ws.checkpoint(f"{task.id}: artifact request failed")
            audit.update(status="failed", error_type=code, task_id=task.id, usage=usage,
                         agent_content=visible_context(audit["events"]), finished=timestamp())
            write_json(self.directory / "audit" / f"{context.request_id}.json", audit)
            self._finish(context.request_id, "failed", None, task.id)
            raise RequestFailure(500, "request_failed") from None


# Expose only bounded requests, public readiness, and operator-authenticated audit reads.
class _Handler(BaseHTTPRequestHandler):
    # Keep default access logging from printing arbitrary client paths and query strings.
    def log_message(self, format, *args) -> None:
        pass

    # Serialize a bounded JSON response without echoing secrets or raw exceptions.
    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # Require the actual loopback Host and a single explicitly scoped bearer credential.
    def _principal(self, *, admin: bool = False) -> str:
        expected_host = self.server.service.url.removeprefix("http://")
        auth = self.headers.get_all("Authorization", [])
        if self.headers.get("Host") != expected_host or len(auth) != 1 or not auth[0].startswith("Bearer "):
            raise RequestFailure(403, "forbidden")
        result = self.server.service.authenticate(auth[0][7:], admin=admin)
        if result is None:
            raise RequestFailure(403, "forbidden")
        return result

    # Return readiness to authenticated clients and private audits only to the operator.
    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            service = self.server.service
            if parsed.path == "/health" and not parsed.query:
                self._principal()
                self._json(200, {"ready": True, "owner_id": service.owner_id,
                                 "case_id": service.config["case_id"]})
            elif parsed.path == "/audit":
                self._principal(admin=True)
                args = parse_qs(parsed.query)
                if set(args) != {"request_id"} or len(args["request_id"]) != 1:
                    raise RequestFailure(400, "invalid_request")
                self._json(200, service.audit(args["request_id"][0]))
            else:
                raise RequestFailure(404, "not_found")
        except RequestFailure as exc:
            self._json(exc.status, {"error": exc.code})
        except Exception:
            self._json(400, {"error": "invalid_request"})

    # Authenticate first, then execute exactly one bounded artifact request without retries.
    def do_POST(self) -> None:
        try:
            if self.path != "/request":
                raise RequestFailure(404, "not_found")
            if self.headers.get("Origin") is not None or self.headers.get("Transfer-Encoding") is not None:
                raise RequestFailure(403, "forbidden")
            principal = self._principal()
            if self.headers.get_content_type() != "application/json":
                raise RequestFailure(415, "json_required")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                raise RequestFailure(411, "length_required")
            size = int(lengths[0])
            if not 1 <= size <= MAX_BODY:
                raise RequestFailure(413, "invalid_size")
            data = json.loads(self.rfile.read(size).decode("utf-8"))
            response = self.server.service.handle_request(principal, data)
            self._json(200, response)
        except RequestFailure as exc:
            self._json(exc.status, {"error": exc.code})
        except (ValueError, UnicodeError):
            self._json(400, {"error": "invalid_request"})
        except Exception:
            self._json(500, {"error": "request_failed"})
