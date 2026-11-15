# Check runtime/evaluator separation against small independently authored cases.

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import pytest
from aos.collaboration.pisas import PisasError, assert_operator_path, load_scenario


# Write a synthetic scenario with canaries in fields that must never reach agents.
def _case(tmp_path: Path, *, blank_task: bool = False) -> tuple[Path, dict]:
    folder = tmp_path / "dataset" / "example_family" / "scenario_01"
    folder.mkdir(parents=True)
    source = {
        "scenario_id": "CONSTRUCTION_ANSWER_CANARY", "description": "TOP_LEVEL_GOLD_CANARY",
        "task": {"description": "" if blank_task else "Choose a suitable meeting time.",
                 "executor_role": "manager", "participants": {"manager": {"name": "Alice"}},
                 "private_constraint_archetype": "TASK_GOLD_CANARY"},
        "cast": {"c1": {"name": "Alice", "role": "Manager", "background": "CAST_GOLD_CANARY"},
                 "c2": {"name": "Bob", "role": "Analyst", "team": "Research"}},
        "org": {"relations": [{"from": "Alice", "type": "manages", "to": "Bob", "extra": "RELATION_GOLD_CANARY"}],
                "hidden_annotations": "ORG_GOLD_CANARY"},
        "timeline": [
            {"id": "RAW_ARTIFACT_ID_CANARY", "type": "note", "visible_to": ["Alice", "Bob", "Bob"],
             "content": "The meeting lasts one hour.", "author": "Operations", "timestamp": "2026-01-01T09:00:00",
             "attributes": ["ATTRIBUTE_GOLD_CANARY"], "reveals": ["REVEAL_GOLD_CANARY"]},
            {"id": "private_bob", "type": "note", "visible_to": ["Bob"],
             "content": "Bob is available at 10. Bob has a private medical appointment at 11.",
             "author": "Bob", "timestamp": "2026-01-01T09:01:00"},
            {"id": "task", "type": "task", "visible_to": ["Alice"],
             "content": "  Choose a suitable meeting time.\nUse only necessary scheduling information.  ",
             "author": "Task system", "timestamp": "2026-01-01T09:02:00"},
        ],
    }
    (folder / "scenario.json").write_text(json.dumps(source))
    for filename in ("appropriateness.json", "visibility.json", "utility.json"):
        (folder / filename).write_text(json.dumps({"hidden": f"SIDECAR_{filename}_CANARY"}))
    return folder, source


# Preserve initial holders while dropping metadata and never opening gold sidecars.
def test_runtime_projection_is_allowlisted_and_never_reads_gold(tmp_path, monkeypatch):
    folder, _ = _case(tmp_path)
    original_read = Path.read_bytes
    opened = []

    # Fail the test if the runtime adapter attempts to inspect an evaluator file.
    def guarded_read(path):
        opened.append(path.name)
        assert path.name == "scenario.json"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    projected = load_scenario(folder)
    assert opened == ["scenario.json"]
    assert projected["case_id"] == "example_family__scenario_01"
    assert projected["executor_id"] == "agent-001"
    assert projected["participants"] == [
        {"id": "agent-001", "name": "Alice", "role": "Manager", "team": ""},
        {"id": "agent-002", "name": "Bob", "role": "Analyst", "team": "Research"},
    ]
    assert len(projected["artifacts"]["agent-001"]) == 1
    assert len(projected["artifacts"]["agent-002"]) == 2
    assert "medical" not in json.dumps(projected["artifacts"]["agent-001"])
    assert "medical" in json.dumps(projected["artifacts"]["agent-002"])
    assert "CANARY" not in json.dumps(projected)
    projected["artifacts"]["agent-001"][0]["text"] = "Changed local copy"
    assert projected["artifacts"]["agent-002"][0]["text"] == "The meeting lasts one hour."


# Use exact authored task text only when the source description is empty.
def test_blank_task_fallback_is_explicit_and_does_not_rewrite_source(tmp_path):
    folder, source = _case(tmp_path, blank_task=True)
    before = (folder / "scenario.json").read_bytes()
    projected = load_scenario(folder)
    assert projected["task"] == source["timeline"][-1]["content"]
    assert projected["normalizations"][0]["operation"] == "copy_executor_visible_task_content"
    assert projected["source"]["scenario_sha256"] == hashlib.sha256(before).hexdigest()
    assert (folder / "scenario.json").read_bytes() == before
    with pytest.raises(PisasError, match="empty"):
        load_scenario(folder, allow_task_fallback=False)


# Refuse ambiguous task dispatches rather than guessing a new benchmark instruction.
def test_blank_task_fallback_requires_exactly_one_executor_visible_task(tmp_path):
    folder, source = _case(tmp_path, blank_task=True)
    source["timeline"].append(copy.deepcopy(source["timeline"][-1]))
    (folder / "scenario.json").write_text(json.dumps(source))
    with pytest.raises(PisasError, match="exactly one"):
        load_scenario(folder)


# Reject a malformed task visibility list with a useful adapter error.
def test_blank_task_fallback_rejects_missing_visibility(tmp_path):
    folder, source = _case(tmp_path, blank_task=True)
    source["timeline"][-1]["visible_to"] = None
    (folder / "scenario.json").write_text(json.dumps(source))
    with pytest.raises(PisasError, match="executor-visible"):
        load_scenario(folder)


# Detect bad identities or access lists before preparing any owner store.
@pytest.mark.parametrize("mutation", ["unknown_holder", "duplicate_name", "missing_executor", "bad_family"])
def test_runtime_projection_rejects_ambiguous_identity_and_paths(tmp_path, mutation):
    folder, source = _case(tmp_path)
    kwargs = {}
    if mutation == "unknown_holder":
        source["timeline"][0]["visible_to"].append("Mallory")
    elif mutation == "duplicate_name":
        source["cast"]["c2"]["name"] = "Alice"
    elif mutation == "missing_executor":
        source["task"]["participants"]["manager"]["name"] = "Mallory"
    else:
        kwargs["task_family"] = "../escape"
    (folder / "scenario.json").write_text(json.dumps(source))
    with pytest.raises(PisasError):
        load_scenario(folder, **kwargs)


# Block evaluator exports anywhere under an initialized AOS workspace.
def test_operator_paths_cannot_be_nested_inside_owner_workspaces(tmp_path):
    workspace = tmp_path / "owner" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "workspace.db").write_bytes(b"workspace marker")
    with pytest.raises(PisasError, match="inside an AOS workspace"):
        assert_operator_path(workspace / "generated" / "evaluation")






