import json
import pytest
from aos.llm import LLM, add_usage, load_prices, parse_json, price_usd


def test_parse_json_handles_fences_and_prose():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('Sure, here it is:\n{"action": "done", "summary": "x"} hope this helps') == {
        "action": "done", "summary": "x"}
    with pytest.raises(json.JSONDecodeError):
        parse_json("not json at all")


def test_prices_and_costs(monkeypatch):
    monkeypatch.delenv("AOS_PRICES", raising=False)
    assert price_usd("gemini-3.6-flash", 1_000_000, 0) == 0.30
    assert price_usd("unknown-model", 10, 10) is None
    monkeypatch.setenv("AOS_PRICES", '{"gemini-3.6-flash": [1, 4], "my-model": [0.5, 0.5]}')
    prices = load_prices()
    assert prices["gemini-3.6-flash"] == (1.0, 4.0) and prices["my-model"] == (0.5, 0.5)
    assert price_usd("my-model", 2_000_000, 2_000_000, prices) == 2.0
    monkeypatch.setenv("AOS_PRICES", "not json")
    with pytest.raises(RuntimeError, match="AOS_PRICES"):
        load_prices()
    total = add_usage({"calls": 1, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.1},
                      None, {"calls": 2, "cost_usd": 0.05, "unpriced_calls": 1})
    assert total == {"calls": 3, "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0, "cost_usd": 0.15, "unpriced_calls": 1}


def test_placeholder_key_is_rejected(monkeypatch):
    monkeypatch.setenv("API_KEY", "your-key")
    monkeypatch.setenv("BASE_URL", "http://127.0.0.1:1/v1")
    with pytest.raises(RuntimeError, match="placeholder"):
        LLM()


def test_llm_records_usage(monkeypatch):
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.delenv("AOS_PRICES", raising=False)
    llm = LLM(model="gemini-3.6-flash")
    before = llm.snapshot()
    llm.record(1_000_000, 100_000, reasoning_tokens=60_000)
    assert llm.usage["calls"] == 1 and llm.usage["cost_usd"] == 0.55 and llm.usage["reasoning_tokens"] == 60_000
    llm.model = "mystery"
    llm.record(10, 10)
    assert llm.usage["unpriced_calls"] == 1 and llm.usage["cost_usd"] == 0.55
    since = llm.usage_since(before)
    assert since["calls"] == 2 and since["input_tokens"] == 1_000_010
    assert llm.available_models() == ["mystery"]  # listing fails offline; the configured model remains
