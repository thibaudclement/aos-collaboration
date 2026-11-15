# Exercise all live dispatch paths against a local scripted provider.

from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from urllib.request import Request, urlopen
import pytest
from aos.collaboration.common import read_json
from aos.collaboration.demo import config
from aos.collaboration.demo.runner import execute_case
from aos.collaboration.demo.store import DemoStore, StoreError
from aos.collaboration.demo.server import CollaborationServer
from aos.collaboration.demo.view import build_view
from aos.collaboration.integrated.contract import CONTRACT_ID
from test_collaboration_service import ALICE_SECRET, BLAIR_SECRET, PUBLIC_ALICE, PUBLIC_BLAIR


# Write two owner-local inputs in the same schema used by the bundled scenarios.
def scenario_file(tmp_path):
    folder = tmp_path / "scheduling" / "scenario_01"
    folder.mkdir(parents=True)
    source = {
        "task": {"description": "Find when Alice and Blair can meet. Report availability only.",
                 "executor_role": "manager", "participants": {"manager": {"name": "Alice"}}},
        "cast": {"a": {"name": "Alice", "role": "Manager", "team": "Operations"},
                 "b": {"name": "Blair", "role": "Analyst", "team": "Operations"}},
        "org": {"relations": []},
        "timeline": [
            {"type": "note", "visible_to": ["Alice"], "content": PUBLIC_ALICE + "\n" + ALICE_SECRET},
            {"type": "note", "visible_to": ["Blair"], "content": PUBLIC_BLAIR + "\n" + BLAIR_SECRET},
        ],
    }
    path = folder / "scenario.json"
    path.write_text(json.dumps(source))
    return path


# Serve OpenAI-compatible response bytes without contacting an external service.
class ProviderHandler(BaseHTTPRequestHandler):
    # Return a controlled response for each actual actor phase.
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system, prompt = request["messages"][0]["content"], request["messages"][1]["content"]
        value = json.loads(prompt)
        phase = system.splitlines()[0].removeprefix("PHASE: ")
        self.server.calls.append({"phase": phase, "system": system, "payload": value})
        if phase == "extract":
            answer = {"claims": [{"text": line, "evidence_ids": [unit["evidence_id"]]}
                                  for unit in value["evidence"] for line in unit["text"].splitlines()]}
        elif phase == "normalize_release":
            answer = {"revisions": [{"parent_claim_id": claim["claim_id"],
                                       "claims": [] if "PRIVATE" in claim["text"] else
                                       [{"text": claim["text"], "evidence_ids": claim["evidence_ids"]}],
                                       "reason": "Private test normalization."}
                                      for claim in value["proposal"]["claims"]]}
        elif phase == "review":
            answer = {"decisions": [{"claim_id": claim["claim_id"], "supported": True,
                                      "appropriate": "PRIVATE" not in claim["text"],
                                      "permitted": "PRIVATE" not in claim["text"], "reason": "Scripted test review."}
                                     for claim in value["proposal"]["claims"]]}
        elif phase == "decide":
            answer = {"final_response": "Meet Tuesday at 10:00.", "used_evidence_refs": value["evidence_refs"],
                      "insufficient_evidence": False}
            if self.server.version == "observed-v2":
                answer.update(decision_action="no_disagreement", decision_note="The released times agree.", disagreements=[])
            if self.server.version == "integrated-v3":
                answer["schedule"] = {"fixture-meeting": {"slot": "Tuesday 10:00", "room": "fixture-room"}}
        else:
            raise AssertionError("Unexpected actor phase")
        self.server.responses.append(answer)
        data = json.dumps({"id": "local-response", "object": "chat.completion", "created": 1,
                           "model": request["model"], "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(answer)}}],
                           "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # Keep request headers and credentials out of test output.
    def log_message(self, *args):
        return


# Verify the real HTTP client, owner services, decision contract, and requester projection together.
@pytest.mark.parametrize("version,count", [("governed-v1", 5), ("observed-v2", 5), ("integrated-v3", 7)])
def test_live_dispatch_with_local_provider(tmp_path, monkeypatch, version, count):
    provider = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
    provider.version, provider.calls, provider.responses = version, [], []
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("API_KEY", "local-test-key")
    monkeypatch.setenv("BASE_URL", f"http://127.0.0.1:{provider.server_port}/v1")
    monkeypatch.delenv("AOS_PRICES", raising=False)
    try:
        result = execute_case(scenario_file(tmp_path), tmp_path / "actor", version=version, model="fixture-model")
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=2)
    assert result["status"] == "completed"
    assert len(provider.calls) == result["usage"]["calls"] == count
    assert result["n_a2a"] == len(result["events"]) == 2
    public = build_view(result, "test-run", "a" * 64)
    for private in (ALICE_SECRET, BLAIR_SECRET):
        assert private not in json.dumps(public)
        assert private not in provider.calls[-1]["payload"]["gathered_info"]
    manifest = read_json(tmp_path / "actor/manifest.json")
    if version == "integrated-v3":
        assert manifest["config"]["top_k"] is None
        assert manifest["config"]["system_version"] == version
        assert result["decision_contract"]["id"] == public["response_contract_id"] == CONTRACT_ID
        assert json.loads(result["final_response"])["schedule"] == provider.responses[-1]["schedule"]
        assert result["decision_audit"][0]["response"] == provider.responses[-1]
        assert [row["phase"] for row in provider.calls] == ["extract", "normalize_release", "review"] * 2 + ["decide"]
    elif version == "observed-v2":
        assert result["decision_observation"]["status"] == "recorded"
        assert public["conflict_detection"]["status"] == "recorded"
    else:
        assert manifest["config"]["top_k"] == 8
        assert "decision_observation" not in result and "response_contract_id" not in public


# Verify that an HTTP start reaches its detached worker and saves a terminal live record.
@pytest.mark.parametrize("version,count", [("governed-v1", 5), ("observed-v2", 5), ("integrated-v3", 7)])
def test_http_start_runs_detached_worker(tmp_path, monkeypatch, version, count):
    provider = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
    provider.version, provider.calls, provider.responses = version, [], []
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    monkeypatch.setenv("API_KEY", "local-worker-test-key")
    monkeypatch.setenv("BASE_URL", f"http://127.0.0.1:{provider.server_port}/v1")
    monkeypatch.delenv("AOS_PRICES", raising=False)
    store = DemoStore(tmp_path / "store", allow_live=True, model="fixture-model")
    case = store.add_case(scenario_file(tmp_path))
    server = CollaborationServer(store, port=0)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        request = Request(server.url + "/api/collaboration/start", method="POST",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps({"case_id": case["case_id"], "version": version,
                                           "launch_id": "worker-test-001"}).encode())
        with urlopen(request, timeout=5) as response:
            receipt = json.load(response)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            snapshot = store.snapshot(receipt["run_id"])
            if snapshot["metadata"]["status"] not in {"queued", "running"}:
                break
            time.sleep(0.1)
        assert snapshot["metadata"]["status"] == "completed"
        assert snapshot["view"]["mode"] == "live"
        assert snapshot["view"]["answer"]["text"]
        assert snapshot["evaluation"]["status"] == "not_loaded"
        assert len(provider.calls) == count
        with urlopen(request, timeout=5) as response:
            duplicate = json.load(response)
        assert duplicate["run_id"] == receipt["run_id"] and duplicate["duplicate"]
        assert len(store.runs()["runs"]) == 1
        assert len(provider.calls) == count
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        store.close()
        for child in store._children:
            child.wait(timeout=5)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=2)


# Keep no-key browsing available while preventing every model-backed start before worker creation.
def test_missing_key_disables_live_without_client_or_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load_env", lambda: None)
    for name in ("API_KEY", "OPENROUTER_API_KEY", "BASE_URL", "AOS_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = config.load_configuration()
    assert settings["live_enabled"] is False
    assert "API_KEY" in settings["disabled_reason"]
    store = DemoStore(tmp_path / "store", allow_live=settings["live_enabled"], disabled_reason=settings["disabled_reason"])
    case = store.add_case(scenario_file(tmp_path))
    monkeypatch.setattr(store, "_launch", lambda *args: pytest.fail("No-key start created a worker"))
    assert store.catalog()["versions"] == ["governed-v1", "observed-v2", "integrated-v3"]
    assert not next(row for row in store.catalog()["cases"] if row["case_id"] == case["case_id"])["live_enabled"]
    for index, version in enumerate(store.catalog()["versions"]):
        with pytest.raises(StoreError, match="Live model execution is disabled"):
            store.start(case["case_id"], version, f"disabled-{index}")
    assert store.runs()["runs"] == []
    store.close()


# Normalize a conventional OpenRouter key alias without revealing it in public settings.
def test_configuration_accepts_alias_and_rejects_placeholders(monkeypatch):
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "local-alias-key")
    settings = config.load_configuration()
    assert settings["live_enabled"] is True
    assert settings["base_url"] == config.DEFAULT_BASE_URL
    assert "local-alias-key" not in json.dumps(settings)
    monkeypatch.setenv("API_KEY", "your-api-key")
    assert config.load_configuration()["live_enabled"] is False
    monkeypatch.setenv("API_KEY", "local-key")
    monkeypatch.setenv("BASE_URL", "https://[")
    assert config.load_configuration()["live_enabled"] is False
