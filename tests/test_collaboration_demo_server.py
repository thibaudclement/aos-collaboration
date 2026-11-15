# HTTP controls for the local collaboration GUI; no provider calls.

import http.client
import json
import os
from pathlib import Path
import threading
import pytest
from aos.collaboration.demo.server import CollaborationServer
from aos.collaboration.demo.store import DemoStore

PRIVATE = "OWNER_PRIVATE_DO_NOT_SERVE"
REF = "ref-" + "a" * 24


# Build a legacy response shape with private fields that must stay outside the API.
def saved_run():
    response = {"request_id": "req-peer", "status": "released", "answer": "The room seats 24.",
                "claims": [{"text": "The room seats 24.", "evidence_refs": [REF]}], "coverage": "not_established"}
    return {"config": {"protocol": "aos-artifact-requests-v1", "mode": "governed", "model": "test-model", "run_index": 0},
            "projection": {"case_id": "sample__scenario_01", "task": "Can 20 people fit?", "executor_id": "agent-001",
                "participants": [{"id": "agent-001", "name": "Alex"}, {"id": "agent-002", "name": "Blair"}],
                "artifacts": {"agent-002": PRIVATE}},
            "status": "completed", "events": [
                {"kind": "request", "sender_id": "agent-001", "recipient_id": "agent-002", "request_id": "req-peer",
                 "body": "Can 20 people fit?", "delivery": "confirmed", "timestamp": "2026-11-01T12:00:00Z"},
                {"kind": "response", "sender_id": "agent-002", "recipient_id": "agent-001", "request_id": "req-peer",
                 "body": json.dumps(response), "delivery": "confirmed", "timestamp": "2026-11-01T12:00:01Z"}],
            "responses": [{"owner_id": "agent-002", "response": response}], "n_a2a": 2,
            "final_response": "Yes, 20 people fit.", "used_evidence_refs": [REF], "insufficient_evidence": False,
            "decision_audit": [{"prompt": PRIVATE}], "agent_contents": {"agent-002": PRIVATE}}


# Start a real loopback HTTP server over a temporary operator registry.
@pytest.fixture
def web(tmp_path):
    store = DemoStore(tmp_path / "operator")
    server = CollaborationServer(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, store
    server.shutdown()
    server.server_close()
    store.close()
    thread.join(timeout=2)


# Exercise exact HTTP request bytes, including Host and Origin controls.
def call(server, path, *, method="GET", payload=None, headers=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    values = dict(headers or {})
    body = raw
    if payload is not None:
        body = json.dumps(payload)
        values.setdefault("Content-Type", "application/json")
    connection.request(method, path, body=body, headers=values)
    response = connection.getresponse()
    data = response.read()
    result = response.status, dict(response.getheaders()), data
    connection.close()
    return result


# Serve the packaged AOS page and deny arbitrary paths into the registry or workspaces.
def test_assets_and_no_raw_file_route(web):
    server, _ = web
    status, headers, data = call(server, "/collaboration")
    assert status == 200 and b"Start a collaboration" in data
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"
    for path in ("/api/file?path=run.json", "/registry.json", "/assets/../../store.py", "/api/collaboration/evaluate"):
        assert call(server, path)[0] == 404


# A link from another site can open the static page without exposing a cross-site data API.
@pytest.mark.parametrize("path", ["/", "/collaboration"])
def test_cross_site_document_navigation_opens_static_page(web, path):
    server, _ = web
    headers = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate",
               "Sec-Fetch-Dest": "document", "Sec-Fetch-User": "?1"}
    status, returned, body = call(server, path, headers=headers)
    assert status == 200 and b"Start a collaboration" in body
    assert returned["Content-Type"] == "text/html; charset=utf-8"
    assert "frame-ancestors 'none'" in returned["Content-Security-Policy"]
    assert "Access-Control-Allow-Origin" not in returned
    # Navigation remains safe when a browser omits the optional user-activation header.
    headers.pop("Sec-Fetch-User")
    assert call(server, path, headers=headers)[0] == 200


# Navigation metadata cannot turn data, assets, frames, commands, or a foreign Origin into page access.
def test_navigation_exception_keeps_cross_site_boundaries(web, monkeypatch):
    server, store = web
    launches = []
    monkeypatch.setattr(store, "_launch", lambda *args: launches.append(args) or os.getpid())
    navigation = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"}
    for path in ("/api/collaboration/catalog", "/api/collaboration/runs", "/api/collaboration/run?id=example",
                 "/assets/collaboration.js", "/assets/collaboration.css"):
        assert call(server, path, headers=navigation)[0] == 403
    for headers in ({"Sec-Fetch-Site": "cross-site"},
                    {**navigation, "Sec-Fetch-Mode": "cors"},
                    {**navigation, "Sec-Fetch-Dest": "iframe"},
                    {**navigation, "Origin": "https://github.com/"},
                    {**navigation, "Origin": "null"},
                    {**navigation, "Host": "attacker.invalid"}):
        assert call(server, "/", headers=headers)[0] == 403
    for path in ("/", "/collaboration", "/api/collaboration/start"):
        status, _, body = call(server, path, method="POST", headers=navigation,
                               payload={"case_id": "availability_demo__scenario_01", "version": "governed-v1",
                                        "launch_id": "launch-navigation"})
        assert status == 403 and json.loads(body)["outcome_unknown"] is False
    assert call(server, server.url + "/", headers=navigation)[0] == 400
    assert launches == []


# Never return private owner audits through the public requester-view endpoint.
def test_requester_snapshot_excludes_private_data(web, tmp_path):
    server, store = web
    source = tmp_path / "run.json"
    source.write_text(json.dumps(saved_run()))
    receipt = store.add_replay(source, title="Saved room example")
    status, _, body = call(server, "/api/collaboration/run?id=" + receipt["run_id"])
    assert status == 200
    assert PRIVATE.encode() not in body
    view = json.loads(body)["view"]
    assert view["answer"]["text"] == "Yes, 20 people fit."
    assert view["conflict_detection"]["status"] == "not_recorded"
    assert call(server, "/api/collaboration/run?id=../../run.json")[0] == 404


# Reject browser requests from another origin before they can start an execution.
@pytest.mark.parametrize("headers", [
    {"Origin": "https://example.invalid"},
    {"Host": "attacker.invalid"},
    {"Sec-Fetch-Site": "cross-site"},
    {"Origin": "null"},
])
def test_origin_and_host_boundaries(web, monkeypatch, headers):
    server, store = web
    launches = []
    monkeypatch.setattr(store, "_launch", lambda *args: launches.append(args) or os.getpid())
    status, _, body = call(server, "/api/collaboration/start", method="POST",
                           payload={"case_id": "availability_demo__scenario_01", "version": "governed-v1", "launch_id": "launch-cross-site"},
                           headers=headers)
    assert status == 403 and json.loads(body)["outcome_unknown"] is False
    assert launches == []


# One launch identity represents one execution even when the start receipt is requested twice.
def test_duplicate_start_is_one_worker(web, monkeypatch):
    server, store = web
    launches = []
    monkeypatch.setattr(store, "_launch", lambda *args: launches.append(args) or os.getpid())
    payload = {"case_id": "availability_demo__scenario_01", "version": "governed-v1", "launch_id": "launch-duplicate"}
    origin = {"Origin": server.url}
    first = call(server, "/api/collaboration/start", method="POST", payload=payload, headers=origin)
    second = call(server, "/api/collaboration/start", method="POST", payload=payload, headers=origin)
    assert first[0] == second[0] == 200
    a, b = json.loads(first[2]), json.loads(second[2])
    assert a["run_id"] == b["run_id"] and a["duplicate"] is False and b["duplicate"] is True
    assert len(launches) == 1


# Reject extra or ambiguous start fields before handing the command to the store.
@pytest.mark.parametrize("raw", [
    '{"case_id":"x","version":"governed-v1","launch_id":"launch-test","path":"/private"}',
    '{"case_id":"x","case_id":"y","version":"governed-v1","launch_id":"launch-test"}',
    '["case","governed-v1","launch-test"]',
])
def test_invalid_launch_contract(web, raw):
    server, _ = web
    status, _, data = call(server, "/api/collaboration/start", method="POST", raw=raw,
                           headers={"Content-Type": "application/json"})
    assert status == 400 and json.loads(data)["outcome_unknown"] is False

