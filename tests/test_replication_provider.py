# Check recorded request settings, uncertain delivery, and provider cost handling offline.

import copy
import json
import pytest
from aos.collaboration.replication import provider


# Return one ordinary OpenAI-compatible actor reply with a recorded OpenRouter charge.
def response(content='{"value": "accepted"}', *, cost=0.012, finish="stop"):
    usage = {"prompt_tokens": 12, "completion_tokens": 5,
             "completion_tokens_details": {"reasoning_tokens": 2}}
    if cost is not None:
        usage["cost"] = cost
    return {"model": "recorded-model", "choices": [{"finish_reason": finish, "message": {"content": content}}],
            "usage": usage}


# Capture attempted request bodies without constructing a real network connection.
class FakeTransport:
    # Store one predetermined reply or exception with the selected provider's cost policy.
    def __init__(self, result, *, base_url=provider.OPENROUTER_URL):
        self.result, self.base_url = result, base_url
        self.cost_required = base_url == provider.OPENROUTER_URL
        self.requests = []

    # Preserve the actual caller body and return a fresh deterministic response.
    def request(self, method, path, body):
        self.requests.append((method, path, copy.deepcopy(body)))
        if isinstance(self.result, BaseException):
            raise self.result
        return copy.deepcopy(self.result)


# Preserve omitted actor parameters for all recorded providers and models.
@pytest.mark.parametrize("model,base_url", [
    ("google/gemini-3.5-flash-lite", provider.OPENROUTER_URL),
    ("anthropic/claude-opus-5.5", provider.OPENROUTER_URL),
    ("gemini-3.5-flash-lite", provider.GOOGLE_URL),
])
def test_actor_request_preserves_original_omissions(tmp_path, model, base_url):
    transport = FakeTransport(response(), base_url=base_url)
    actor = provider.MeteredModel(provider.Budget(transport), tmp_path, model, max_calls=1, progress=lambda message: None)
    assert actor.complete_json("question", system="instruction") == {"value": "accepted"}
    request = transport.requests[0][2]
    assert request == {"model": model,
                       "messages": [{"role": "system", "content": "instruction"}, {"role": "user", "content": "question"}],
                       "response_format": {"type": "json_object"}}
    receipt = json.loads((tmp_path / "calls/001.json").read_text())
    assert receipt["request"] == request
    assert receipt["response"] == response()
    assert receipt["provider_cost_usd"] == 0.012
    assert actor.usage["reasoning_tokens"] == 2
    with pytest.raises(provider.ProviderError):
        actor.complete_json("question", system="instruction")
    assert len(transport.requests) == 1


# Retain missing direct-Google prices without inventing a zero provider charge.
def test_google_unreported_cost_is_explicit(tmp_path):
    transport = FakeTransport(response(cost=None), base_url=provider.GOOGLE_URL)
    actor = provider.MeteredModel(provider.Budget(transport), tmp_path, "gemini-3.5-flash-lite", max_calls=1, progress=lambda message: None)
    assert actor.complete_json("question", system="instruction")
    receipt = json.loads((tmp_path / "calls/001.json").read_text())
    assert receipt["status"] == "ok"
    assert receipt["provider_cost_usd"] is None
    assert receipt["cost_status"] == "not_reported_by_provider"
    assert actor.usage["unpriced_calls"] == 1


# Stop uncertain dispatches once while preserving the charge's unknown status.
def test_actor_transport_failure_is_not_retried(tmp_path):
    transport = FakeTransport(TimeoutError("private provider text"))
    actor = provider.MeteredModel(provider.Budget(transport), tmp_path, "google/gemini-3.5-flash-lite", max_calls=5, progress=lambda message: None)
    for _ in range(2):
        with pytest.raises(provider.ProviderError) as caught:
            actor.complete_json("question", system="instruction")
        assert "private provider text" not in str(caught.value)
    assert len(transport.requests) == 1
    receipt = json.loads((tmp_path / "calls/001.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["provider_cost_usd"] is None
    assert receipt["error_type"] == "TimeoutError"


# Preserve a charged malformed response without converting it into a successful actor answer.
def test_actor_invalid_json_keeps_cost_and_raw_reply(tmp_path):
    transport = FakeTransport(response("not a JSON response"))
    actor = provider.MeteredModel(provider.Budget(transport), tmp_path, "google/gemini-3.5-flash-lite", max_calls=3, progress=lambda message: None)
    with pytest.raises(provider.ProviderError):
        actor.complete_json("question", system="instruction")
    receipt = json.loads((tmp_path / "calls/001.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["response"]["choices"][0]["message"]["content"] == "not a JSON response"
    assert receipt["provider_cost_usd"] == actor.usage["cost_usd"] == 0.012
    assert actor.usage["unpriced_calls"] == 0


# Reject missing OpenRouter prices after one received response without sending another request.
def test_openrouter_unknown_cost_stops_actor(tmp_path):
    transport = FakeTransport(response(cost=None))
    actor = provider.MeteredModel(provider.Budget(transport), tmp_path, "google/gemini-3.5-flash-lite", max_calls=3, progress=lambda message: None)
    with pytest.raises(provider.ProviderError):
        actor.complete_json("question", system="instruction")
    assert len(transport.requests) == 1
    receipt = json.loads((tmp_path / "calls/001.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["response"] == response(cost=None)


# Refuse an arbitrary provider origin before any transport can be used.
def test_transport_rejects_changed_endpoint():
    with pytest.raises(provider.ProviderError):
        provider.Transport("fixture-key", base_url="https://other.example/api/v1")


# Keep receipt creation exclusive and leave original bytes intact after a duplicate attempt.
def test_receipts_are_exclusive_and_private(tmp_path):
    path = tmp_path / "receipt.json"
    provider.write_record(path, {"status": "started"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        provider.write_record(path, {"status": "replacement"})
    assert path.read_bytes() == original
    assert path.stat().st_mode & 0o777 == 0o600
