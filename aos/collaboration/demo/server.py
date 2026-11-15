# Serve the local collaboration view without exposing operator files or owner audits.

from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from .store import DemoStore, StoreError
from .config import load_configuration

ASSETS = {
    "/": ("collaboration.html", "text/html; charset=utf-8"),
    "/collaboration": ("collaboration.html", "text/html; charset=utf-8"),
    "/assets/collaboration.css": ("collaboration.css", "text/css; charset=utf-8"),
    "/assets/collaboration.js": ("collaboration.js", "text/javascript; charset=utf-8"),
}
STATIC = Path(__file__).resolve().parents[2] / "static"
MAX_POST = 8192


# Reject ambiguous JSON objects before accepting a launch request.
def unique_object(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("Duplicate request field")
        value[key] = item
    return value


# Restrict the collaboration HTTP surface to explicit view and launch operations.
class CollaborationHandler(BaseHTTPRequestHandler):
    # Validate the loopback authority and prevent cross-origin browser access.
    def _local_request(self):
        host = self.headers.get("Host", "")
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update({"127.0.0.1", "localhost"})
        if host not in allowed:
            self._json({"error": "Unexpected local server address.", "outcome_unknown": False}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + host:
            self._json({"error": "Use this server's own collaboration page.", "outcome_unknown": False}, 403)
            return False
        url = urlsplit(self.path)
        if url.scheme or url.netloc:
            self._json({"error": "Absolute request targets are not accepted.", "outcome_unknown": False}, 400)
            return False
        # External links may open the static page; APIs, embedded assets, and commands stay protected.
        page_navigation = (self.command == "GET" and url.path in {"/", "/collaboration"}
                           and self.headers.get("Sec-Fetch-Mode") == "navigate"
                           and self.headers.get("Sec-Fetch-Dest") == "document")
        if self.headers.get("Sec-Fetch-Site") == "cross-site" and not page_navigation:
            self._json({"error": "Cross-site requests are not accepted.", "outcome_unknown": False}, 403)
            return False
        return True

    # Read only the catalogue, public run summaries, or the filtered requester view.
    def do_GET(self):
        if not self._local_request():
            return
        url = urlsplit(self.path)
        try:
            if url.path in ASSETS:
                filename, content_type = ASSETS[url.path]
                self._send(200, (STATIC / filename).read_bytes(), content_type)
            elif url.path == "/api/collaboration/catalog":
                self._json(self.server.store.catalog())
            elif url.path == "/api/collaboration/runs":
                self._json(self.server.store.runs())
            elif url.path == "/api/collaboration/run":
                params = parse_qs(url.query, keep_blank_values=True)
                if set(params) != {"id"} or len(params["id"]) != 1:
                    raise ValueError("Supply one conversation identifier.")
                self._json(self.server.store.snapshot(params["id"][0]))
            else:
                self._json({"error": "Not found."}, 404)
        except StoreError as exc:
            self._json({"error": str(exc)}, getattr(exc, "status", 400))
        except FileNotFoundError:
            self._json({"error": "Conversation not found."}, 404)
        except (ValueError, KeyError):
            self._json({"error": "Invalid conversation request."}, 400)
        except Exception:
            self._json({"error": "The local view is temporarily unavailable. The last displayed record is unchanged."}, 500)

    # Accept one idempotent launch command (evaluation and arbitrary file operations are unavailable).
    def do_POST(self):
        if not self._local_request():
            return
        if urlsplit(self.path).path != "/api/collaboration/start":
            self._json({"error": "Not found.", "outcome_unknown": False}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_POST or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Expected a small JSON launch request.")
            body = json.loads(self.rfile.read(size), object_pairs_hook=unique_object)
            if not isinstance(body, dict) or set(body) != {"case_id", "version", "launch_id"}:
                raise ValueError("Unexpected launch request fields.")
            if any(not isinstance(value, str) or not value or len(value) > 200 for value in body.values()):
                raise ValueError("Invalid launch request values.")
        except (ValueError, UnicodeError, json.JSONDecodeError):
            self._json({"error": "Invalid launch request.", "outcome_unknown": False}, 400)
            return
        try:
            receipt = self.server.store.start(body["case_id"], body["version"], body["launch_id"])
            self._json(receipt, 200)
        except StoreError as exc:
            self._json({"error": str(exc), "outcome_unknown": bool(exc.outcome_unknown)}, getattr(exc, "status", 400))
        except Exception:
            self._json({"error": "The launch receipt is unavailable. Check conversation history before checking the same request again.",
                        "outcome_unknown": True}, 500)

    # Return JSON as data rather than executing model-generated markup.
    def _json(self, value, status=200):
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # Apply conservative browser headers on both successful responses and errors.
    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    # Keep repetitive polling logs out of the terminal while the backend records execution.
    def log_message(self, format, *args):
        return


# Bind only to loopback and hold the already-filtering store behind the HTTP handler.
class CollaborationServer(ThreadingHTTPServer):
    daemon_threads = True
    
    # Reuse the same store across requests so starts and history remain consistent.
    def __init__(self, store, host="127.0.0.1", port=8770):
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("The collaboration UI is a local application; bind to 127.0.0.1.")
        self.store = store
        super().__init__((host, port), CollaborationHandler)

    # Report the actual bound port, including ephemeral ports used by tests
    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"


# Start the AOS collaboration interface without constructing a model for replay.
def serve(root, host="127.0.0.1", port=8770, *, allow_live=False,
          model=None, top_k=8):
    configuration = load_configuration(model)
    allow_live = bool(allow_live and configuration["live_enabled"])
    store = DemoStore(root, allow_live=allow_live, model=configuration["model"], top_k=top_k,
                      disabled_reason=configuration["disabled_reason"] or "Restart with live execution enabled to start a new collaboration.")
    server = CollaborationServer(store, host, port)
    print(f"AOS collaboration: {server.url}/collaboration", flush=True)
    print("Recorded replay is available. " +
          ("Live model execution is enabled; no official judge is launched by this UI."
           if allow_live else "Live model execution is disabled."), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping the local collaboration interface.", flush=True)
    finally:
        server.server_close()
        store.close()
