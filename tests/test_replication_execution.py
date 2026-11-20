# Exercise replication stages through the real actor services and pinned evaluator offline.

from copy import deepcopy
import hashlib
import http.client
import json
import shutil
import time

import pytest

from aos.collaboration.integrated import contract
from aos.collaboration.replication import native, provider, runner
from test_replication_native import judgment_reply, response, scenario


# Block provider connections while allowing the real owner services to use localhost HTTP.
@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Stop a real HTTPS connection before any provider bytes can leave the process.
    def denied(*args, **kwargs):
        raise AssertionError("Provider access is forbidden in offline tests")

    monkeypatch.setattr(http.client.HTTPSConnection, "connect", denied)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(native, "JUDGE_GAP_SECONDS", 0)


# Answer real runtime requests with deterministic fixture responses at the provider boundary.
class FixtureTransport:
    # Select one fixed actor treatment or the official evaluation fixture.
    def __init__(self, *, model, contract_enabled=False, native_actor=False, judging=False):
        self.model, self.contract_enabled = model, contract_enabled
        self.native_actor, self.judging = native_actor, judging
        self.base_url = provider.GOOGLE_URL if model == "gemini-3.5-flash-lite" else provider.OPENROUTER_URL
        self.cost_required = self.base_url == provider.OPENROUTER_URL
        self.posts, self.phases, self.timeout = [], [], 90

    # Return fixture JSON while recording the exact request sent by the production adapter.
    def request(self, method, path, body):
        assert (method, path) == ("POST", "/api/v1/chat/completions")
        self.posts.append(deepcopy(body))
        if self.judging:
            return judgment_reply(body)
        if self.native_actor:
            return response("FINAL: Room Alder seats 24 people." if len(self.posts) == 1 else "ready")
        assert body["model"] == self.model
        assert set(body) == {"model", "messages", "response_format"}
        phase = body["messages"][0]["content"].splitlines()[0].removeprefix("PHASE: ")
        payload = json.loads(body["messages"][1]["content"])
        self.phases.append(phase)
        if phase == "extract":
            value = {"claims": [{"text": unit["text"], "evidence_ids": [unit["evidence_id"]]}
                                for unit in payload["evidence"]]}
        elif phase == "normalize_release":
            value = {"revisions": [{"parent_claim_id": item["claim_id"],
                                    "claims": [{"text": item["text"], "evidence_ids": item["evidence_ids"]}],
                                    "reason": "Preserved supported room capacity."}
                                   for item in payload["proposal"]["claims"]]}
        elif phase == "review":
            value = {"decisions": [{"claim_id": item["claim_id"], "supported": True,
                                    "appropriate": True, "permitted": True,
                                    "reason": "The shared capacity fact supports this task."}
                                   for item in payload["proposal"]["claims"]]}
        elif phase == "decide":
            value = {"final_response": "ready", "used_evidence_refs": payload["evidence_refs"],
                     "insufficient_evidence": False}
            if self.contract_enabled:
                value["decision"] = "ready"
        else:
            raise AssertionError("Unexpected fixture model phase")
        raw = response(json.dumps(value))
        if not self.cost_required:
            raw["usage"].pop("cost")
        return raw


# Prepare isolated scenario and evaluator directories with a single authored slot.
def prepared(tmp_path, phase, arm, model, response_contract, run_index):
    root = tmp_path / "batch"
    labels = scenario(root / "inputs")
    case = "synthetic/scenario_01"
    actor = root / "actor-inputs" / case / "scenario.json"
    actor.parent.mkdir(parents=True)
    shutil.copyfile(labels / "scenario.json", actor)
    selected = {"case": case, "owner_count": 2,
                "actor_scenario": actor.relative_to(root).as_posix(),
                "actor_sha256": hashlib.sha256(actor.read_bytes()).hexdigest(),
                "judge_input": labels.relative_to(root).as_posix()}
    run = {"case": case, "arm": arm, "run_index": run_index,
           "path": f"slots/{case}/{arm}/run{run_index}", "model": model,
           "base_url": provider.GOOGLE_URL if model == "gemini-3.5-flash-lite" else provider.OPENROUTER_URL,
           "key_env": "GOOGLE_API_KEY" if model == "gemini-3.5-flash-lite" else "API_KEY",
           "response_contract": response_contract,
           "version": {"native": "native-decentralized", "governed-v1": "governed-v1",
                       "integrated": "integrated-v3"}[arm],
           "maximum_calls": 32 if arm == "native" else 7 if arm == "integrated" else 5}
    generation = {**run, "id": "fixture:generate", "stage": "generate"}
    evaluation = {**run, "id": "fixture:judge", "stage": "judge", "maximum_calls": 100,
                  "key_env": "API_KEY", "base_url": provider.OPENROUTER_URL}
    plan = {"phase": phase, "inputs": [selected], "stages": [generation, evaluation]}
    return root, plan, generation, evaluation


# Run every distinct actor/provider/contract route through generation and its bound evaluation.
@pytest.mark.parametrize("phase,arm,model,response_contract,run_index", [
    ("external-baseline", "governed-v1", "gemini-3.5-flash-lite", False, 1),
    ("calibration", "governed-v1", native.ACTOR_MODEL, False, 0),
    ("wave2", "governed-v1", native.ACTOR_MODEL, True, 0),
    ("wave2", "integrated", native.ACTOR_MODEL, True, 0),
    ("wave3", "integrated", native.OPUS_MODEL, True, 0),
    ("external-baseline", "native", native.ACTOR_MODEL, False, 1),
    ("wave3", "native", native.OPUS_MODEL, False, 0),
])
def test_real_generation_and_judgment_stages(tmp_path, phase, arm, model, response_contract, run_index):
    root, plan, generation, evaluation = prepared(tmp_path, phase, arm, model, response_contract, run_index)
    channel = FixtureTransport(model=model, contract_enabled=response_contract, native_actor=arm == "native")
    generated = runner.attempt_stage(root, plan, generation, provider.Budget(channel), lambda _: None)
    assert generated["status"] == "ok", generated
    assert generated["model_calls"] == len(channel.posts)
    assert runner.stage_state(root, plan, generation)["status"] == "ok"
    directory = root / generation["path"]
    pipeline = runner.read_object(directory / "pipeline.json")
    assert pipeline["config"]["agent_llm"] == model
    assert pipeline["config"]["run_index"] == run_index
    assert "EVALUATOR_" not in json.dumps(channel.posts)
    if arm == "native":
        assert pipeline["config"]["scenario_dir"] == str(root / "inputs/synthetic/scenario_01")
        assert generated["model_calls"] == 2
    else:
        recorded = runner.read_object(directory / "actor/run.json")
        assert recorded["status"] == "completed" and recorded["degraded"] is False
        assert pipeline["config"]["source"]["scenario_sha256"] == plan["inputs"][0]["actor_sha256"]
        assert pipeline["final_response"] == recorded["final_response"]
        assert pipeline["n_a2a"] == len(pipeline["log"]) == 2
        assert channel.phases == (["extract", "normalize_release", "review"] * 2 + ["decide"]
                                   if arm == "integrated" else ["extract", "review"] * 2 + ["decide"])
        if response_contract:
            assert recorded["config"]["response_contract"] == contract.specification()
            assert pipeline["config"]["response_contract"] == contract.specification()
            assert recorded["decision_contract"]["operation"] == "serialize_answer_fields"
            assert json.loads(pipeline["final_response"]) == {"final_response": "ready", "decision": "ready"}
        else:
            assert "response_contract" not in pipeline["config"]
            assert pipeline["final_response"] == "ready"
    if model == "gemini-3.5-flash-lite":
        assert generated["known_response_cost_usd"] == 0
        assert generated["unreported_cost_calls"] == generated["model_calls"]
    evaluation_channel = FixtureTransport(model=native.ACTOR_MODEL, judging=True)
    evaluated = runner.attempt_stage(root, plan, evaluation, provider.Budget(evaluation_channel), lambda _: None)
    assert evaluated["status"] == "ok", evaluated
    assert evaluated["evaluated_pipeline_sha256"] == runner.file_hash(directory / "pipeline.json")
    assert runner.stage_state(root, plan, evaluation)["status"] == "ok"
    judgment = runner.read_object(directory / "judgment.json")
    assert judgment["config"]["agent_llm"] == model
    assert judgment["utility"]["correctness"] is True
    assert judgment["verification"]["degraded"] is False
    assert set(evaluated["call_receipt_sha256"]) == {
        path.relative_to(root).as_posix() for path in (directory / "judging/calls").glob("*.json")}
