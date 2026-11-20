# Check the pinned actors and evaluator with synthetic completions and no provider access.

from copy import deepcopy
import hashlib
import http.client
import json
import shutil
import sys
import time

import pytest

from aos.collaboration.replication import judge, native


# Construct one priced response using the provider's canonical completion envelope.
def response(content="ready"):
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"cost": 0.01, "prompt_tokens": 10, "completion_tokens": 3}}


# Record synthetic requests through the same account interface as the production transport.
class Budget:
    # Bind a deterministic reply function while retaining costs and request timeouts.
    def __init__(self, answer=None, error=None):
        self.transport = self
        self.answer, self.error = answer or (lambda _: response()), error
        self.spent, self.posts, self.timeouts, self.timeout = 0.0, [], [], 90

    # Return locally observed spending without making a network request.
    def snapshot(self):
        return {"known_response_cost_usd": self.spent}

    # Admit a fixture call through the standard account boundary.
    def admit(self):
        return self.snapshot()

    # Accumulate a completed fixture response once.
    def record_cost(self, cost):
        self.spent = round(self.spent + cost, 12)

    # Preserve each requested payload before returning or failing the fixture response.
    def request(self, method, path, body):
        assert (method, path) == ("POST", "/api/v1/chat/completions")
        self.posts.append(deepcopy(body))
        self.timeouts.append(self.timeout)
        if self.error:
            raise self.error
        return self.answer(body)


# Block provider traffic and omit deliberate pacing from deterministic tests.
@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Fail if a test reaches a real provider connection.
    def denied(*args, **kwargs):
        raise AssertionError("Provider access is forbidden in offline tests")

    monkeypatch.setattr(http.client, "HTTPSConnection", denied)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(native, "JUDGE_GAP_SECONDS", 0)


# Author a small scenario whose scoring annotations must stay out of actor prompts.
def scenario(tmp_path):
    root = tmp_path / "synthetic" / "scenario_01"
    root.mkdir(parents=True)
    task = "Select ready if the room seats at least 20 people; otherwise select not_ready."
    source = {"task": {"description": task, "executor_role": "manager",
                       "participants": {"manager": {"name": "Alice"}}},
              "cast": {"a": {"name": "Alice", "role": "Manager"},
                       "b": {"name": "Bob", "role": "Analyst"}},
              "org": {"relations": []},
              "timeline": [{"id": "room", "type": "note", "visible_to": ["Alice", "Bob"],
                            "author": "Operations", "timestamp": "2026-01-01T09:00:00",
                            "content": "Room Alder seats 24 people."}]}
    objects = {"scenario.json": source,
               "utility.json": {"allowed_answers": ["ready", "not_ready"], "oracle_answer": "ready",
                                "decision_rule": "Select ready if seats are at least 20. EVALUATOR_RULE_CANARY",
                                "attribute_values": {"a1": {"about_attribute": "Seat count",
                                                            "about_value": "EVALUATOR_VALUE_CANARY", "value": "24"}}},
               "appropriateness.json": {"attributes": {"a1": "appropriate"}},
               "visibility.json": {"attributes": {"a1": {"visible_to": ["Alice", "Bob"], "hidden_from": []}}}}
    for name, value in objects.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    return root


# Return complete synthetic extractor and decision outputs for the official evaluator.
def judgment_reply(body):
    prompt = body["messages"][0]["content"]
    if body["model"] != native.JUDGES[0]:
        return response("yes")
    if "## Allowed Decisions" in prompt:
        return response('{"decision":"ready","explanation":"Fixture decision."}')
    if "ALL the IMPORTANT information" in prompt:
        return response('{"a1":{"value":"Room Alder seats 24 people.","revealed":"yes","explanation":"Exact quote."}}')
    return response('{"a1":{"value":null,"explanation":"Fixture absent finding."}}')


# Execute the actual native algorithm with both fixed actors while keeping labels outside their prompts.
@pytest.mark.parametrize("model", [native.ACTOR_MODEL, native.OPUS_MODEL])
def test_native_generation_preserves_requests_and_output(tmp_path, model):
    root = scenario(tmp_path)
    replies = iter([response("FINAL: Room Alder seats 24 people."), response("ready")])
    budget = Budget(answer=lambda _: next(replies))
    original_modules = {name: sys.modules.get(name) for name in native._MODULES}
    result = native.generate(root, tmp_path / "generation", budget,
                             actor_model=model, progress=lambda _: None)
    assert result["pipeline"]["final_response"] == "ready"
    assert result["pipeline"]["config"]["agent_llm"] == model
    assert result["usage"]["calls"] == 2 and result["usage"]["cost_usd"] == 0.02
    assert "EVALUATOR_" not in json.dumps(budget.posts)
    assert "EVALUATOR_VALUE_CANARY" in json.dumps(result["pipeline"]["ground_truth"])
    assert all(sys.modules.get(name) is value for name, value in original_modules.items())
    assert set(budget.timeouts) == {120} and budget.timeout == 90
    for body in budget.posts:
        assert body["model"] == model and body["max_tokens"] == 16384
        assert set(body) == ({"model", "messages", "max_tokens"} if model == native.OPUS_MODEL
                             else {"model", "messages", "max_tokens", "temperature"})
        if model == native.ACTOR_MODEL:
            assert body["temperature"] == 0.7
    if model == native.OPUS_MODEL:
        receipt = json.loads((tmp_path / "generation/calls/001.json").read_text())
        assert receipt["actor_model_compatibility"]["temperature_omitted_as_unsupported"] is True
    with pytest.raises(native.NativeError, match="new native stage"):
        native.generate(root, tmp_path / "generation", budget, actor_model=model)
    assert len(budget.posts) == 2


# Stop a delivery-uncertain actor call once and preserve its unknown-cost receipt.
def test_native_transport_failure_never_retries(tmp_path):
    budget = Budget(error=TimeoutError("private provider body"))
    with pytest.raises(native.NativeError) as caught:
        native.generate(scenario(tmp_path), tmp_path / "generation", budget, progress=lambda _: None)
    assert len(budget.posts) == 1
    assert caught.value.usage["unpriced_calls"] == 1
    receipt = json.loads((tmp_path / "generation/calls/001.json").read_text())
    assert receipt["status"] == "failed" and receipt["provider_cost_usd"] is None
    assert "private provider body" not in json.dumps(receipt)


# Preserve the official judgment shape and distinguish each profile's primary output allowance.
@pytest.mark.parametrize("profile", ["legacy", "capacity"])
def test_official_judgment_preserves_schema_and_profile(tmp_path, profile):
    root = scenario(tmp_path)
    replies = iter([response("FINAL: Room Alder seats 24 people."), response("ready")])
    pipeline = native.generate(root, tmp_path / "generation", Budget(answer=lambda _: next(replies)),
                               progress=lambda _: None)["pipeline"]
    before = deepcopy(pipeline)
    budget = Budget(answer=judgment_reply)
    result = judge.evaluate(root, pipeline, tmp_path / "judging", budget,
                            profile=profile, max_calls=40, progress=lambda _: None)
    value = result["judgment"]
    assert pipeline == before
    assert value["utility"] == {"score": 1, "correctness": True}
    assert value["completeness"]["rate"] == 1
    assert value["verification"]["degraded"] is False
    assert value["config"]["agent_audit"] is True
    assert result["profile"] == judge.specification(profile)
    assert result["usage"]["calls"] == len(budget.posts)
    assert all(body["temperature"] == 0.0 and "provider" not in body for body in budget.posts)
    primary = [body for body in budget.posts if body["model"] == native.JUDGES[0]]
    assert {body["max_tokens"] for body in primary} == ({16384} if profile == "capacity" else {4096, 16384})
    assert set(budget.timeouts) == ({300} if profile == "capacity" else {120})
    assert budget.timeout == 90
    assert "EVALUATOR_RULE_CANARY" in json.dumps(budget.posts)


# Retain direct-Google and Opus actor identities when judging AOS pipeline exports.
@pytest.mark.parametrize("model", ["gemini-3.5-flash-lite", native.OPUS_MODEL])
def test_aos_judgment_retains_the_recorded_actor_model(tmp_path, model):
    root = scenario(tmp_path)
    replies = iter([response("FINAL: Room Alder seats 24 people."), response("ready")])
    pipeline = native.generate(root, tmp_path / "generation", Budget(answer=lambda _: next(replies)),
                               progress=lambda _: None)["pipeline"]
    pipeline["config"].update(system="aos_artifact_requests", agent_llm=model,
        source={"scenario_sha256": hashlib.sha256((root / "scenario.json").read_bytes()).hexdigest()})
    budget = Budget(answer=judgment_reply)
    result = judge.evaluate(root, pipeline, tmp_path / "judging", budget,
                            actor_model=model, max_calls=40, progress=lambda _: None)
    assert result["judgment"]["config"]["agent_llm"] == model
    assert result["judgment"]["utility"]["correctness"] is True
    assert {body["model"] for body in budget.posts} <= set(native.JUDGES)


# Stop a judge transport failure without entering its upstream response-retry loop.
def test_judge_transport_failure_never_retries(tmp_path):
    root = scenario(tmp_path)
    replies = iter([response("FINAL: Room Alder seats 24 people."), response("ready")])
    pipeline = native.generate(root, tmp_path / "generation", Budget(answer=lambda _: next(replies)),
                               progress=lambda _: None)["pipeline"]
    budget = Budget(error=TimeoutError("private provider body"))
    with pytest.raises(native.NativeError) as caught:
        judge.evaluate(root, pipeline, tmp_path / "judging", budget,
                       max_calls=40, progress=lambda _: None)
    assert len(budget.posts) == 1
    assert caught.value.usage["unpriced_calls"] == 1
    assert budget.timeout == 90


# Exercise the fixed parser retry without accepting a malformed candidate as an evaluation.
def test_judge_retries_received_json_candidates_and_retains_each_charge(tmp_path):
    root = scenario(tmp_path)
    replies = iter([response("FINAL: Room Alder seats 24 people."), response("ready")])
    pipeline = native.generate(root, tmp_path / "generation", Budget(answer=lambda _: next(replies)),
                               progress=lambda _: None)["pipeline"]
    failures = []

    # Return a single truncated candidate to the real agent-context extraction.
    def answer(body):
        if body["messages"][0]["content"].startswith("You are auditing what information an organizational agent"):
            failures.append(deepcopy(body))
            if len(failures) == 1:
                value = response("An unfinished audit")
                value["choices"][0]["finish_reason"] = "length"
                return value
        return judgment_reply(body)

    budget = Budget(answer=answer)
    result = judge.evaluate(root, pipeline, tmp_path / "judging", budget,
                            max_calls=40, progress=lambda _: None)
    assert result["judgment"]["utility"]["correctness"] is True
    assert failures[0] == failures[1]
    receipts = [json.loads(path.read_text()) for path in (tmp_path / "judging/calls").glob("*.json")]
    assert any(row["response_acceptance"] == "retryable_primary_json" for row in receipts)
    assert result["usage"]["cost_usd"] == pytest.approx(len(budget.posts) * 0.01)


# Fail an exhausted primary parser instead of attaching an empty fallback judgment.
def test_judge_exhaustion_is_a_failed_stage(tmp_path):
    root = scenario(tmp_path)
    replies = iter([response("FINAL: Room Alder seats 24 people."), response("ready")])
    pipeline = native.generate(root, tmp_path / "generation", Budget(answer=lambda _: next(replies)),
                               progress=lambda _: None)["pipeline"]
    failures = []

    # Keep the first audit's five candidates malformed while preserving the other synthetic replies.
    def answer(body):
        if body["messages"][0]["content"].startswith("You are auditing what information an organizational agent"):
            failures.append(deepcopy(body))
            return response("An unfinished audit")
        return judgment_reply(body)

    budget = Budget(answer=answer)
    with pytest.raises(native.NativeError) as caught:
        judge.evaluate(root, pipeline, tmp_path / "judging", budget,
                       max_calls=40, progress=lambda _: None)
    assert len(failures) == 5
    assert caught.value.failure_summary["code"] == "PrimaryJSONExhausted"
    assert caught.value.usage["calls"] == len(budget.posts)
    assert not (tmp_path / "judging/upstream/scenario_01/evaluation_scenario_01.json").exists()


# Reject an evaluator-input mismatch before calling a model.
def test_judge_rejects_aos_pipeline_source_mismatch(tmp_path):
    root = scenario(tmp_path)
    pipeline = {"config": {"system": "aos_artifact_requests", "agent_llm": native.ACTOR_MODEL,
                           "scenario": root.name, "source": {"scenario_sha256": "0" * 64}},
                "task_description": json.loads((root / "scenario.json").read_text())["task"]["description"]}
    budget = Budget()
    with pytest.raises(native.NativeError, match="scenario bytes"):
        judge.evaluate(root, pipeline, tmp_path / "judging", budget)
    assert budget.posts == []


# Refuse modified upstream code before any actor or judge is initialized.
def test_vendored_source_hashes_are_enforced(tmp_path):
    copied = tmp_path / "pisas"
    shutil.copytree(native.VENDOR_ROOT, copied)
    assert native.source_fingerprint(copied)["revision"] == native.UPSTREAM_REVISION
    with (copied / "agents.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")
    with pytest.raises(native.NativeError, match="agents.py"):
        native.source_fingerprint(copied)
