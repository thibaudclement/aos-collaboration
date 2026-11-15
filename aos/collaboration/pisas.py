# Project PiSAs runtime inputs without reading evaluator annotations.
# The source's ``visible_to`` grants initial artifact access, not blanket
# permission to disclose its contents to another participant.

from __future__ import annotations
import copy
import hashlib
import json
import re
from pathlib import Path

PISAS_CODE_REVISION = "1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd"
PISAS_DATA_REVISION = "8a1ba30ba210d4a430bbd25cf4a3500d0e22c5b9"
PISAS_FILES = ("scenario.json", "appropriateness.json", "visibility.json", "utility.json")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}\Z")


# Report malformed source data without accidentally loading evaluator annotations.
class PisasError(ValueError):
    pass


# Validate one file-system component used in run and evaluator output paths.
def _component(value: object, field: str) -> str:
    if not isinstance(value, str) or not _PATH_COMPONENT.fullmatch(value):
        raise PisasError(f"{field} must contain only letters, digits, underscores, and hyphens")
    return value


# Require a text value while preserving the exact source wording.
def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise PisasError(f"{field} must be {'a string' if allow_empty else 'nonempty text'}")
    return value


# Require a dictionary where the upstream schema defines a named object.
def _object(value: object, field: str) -> dict:
    if not isinstance(value, dict):
        raise PisasError(f"{field} must be a JSON object")
    return value


# Hash exactly the source bytes recorded in the reproducibility manifest.
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Keep evaluator materials and exports outside every initialized AOS workspace.
def assert_operator_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    for ancestor in (resolved, *resolved.parents):
        if (ancestor / "workspace.db").exists():
            raise PisasError(f"Operator-only files cannot be written inside an AOS workspace: {ancestor}")
    return resolved


# Resolve a source folder or scenario file without reading any neighboring labels.
def _scenario_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    candidate = candidate / "scenario.json" if candidate.is_dir() else candidate
    if candidate.name != "scenario.json" or not candidate.is_file():
        raise PisasError("Expected a scenario folder or its scenario.json file")
    return candidate


# Select the authored task, recording the explicit fallback used by older bundles.
def _task_text(source: dict, executor_name: str, *, allow_task_fallback: bool) -> tuple[str, list[dict]]:
    task = _object(source.get("task"), "task")
    description = _text(task.get("description", ""), "task.description", allow_empty=True)
    if description.strip():
        return description, []
    if not allow_task_fallback:
        raise PisasError("task.description is empty; enable the recorded executor-visible task fallback")
    candidates = [entry for entry in source["timeline"]
                  if entry.get("type") == "task" and isinstance(entry.get("visible_to"), list)
                  and executor_name in entry["visible_to"]
                  and isinstance(entry.get("content"), str) and entry["content"].strip()]
    if len(candidates) != 1:
        raise PisasError("Empty task.description needs exactly one nonempty executor-visible task entry")
    return candidates[0]["content"], [{
        "field": "task.description", "operation": "copy_executor_visible_task_content",
        "reason": "The source task.description is empty.",
    }]


# Build a runtime-only projection of a PiSAs scenario with initial access preserved.
def load_scenario(path: str | Path, *, task_family: str | None = None,
                  allow_task_fallback: bool = True) -> dict:
    source_path = _scenario_file(path)
    raw = source_path.read_bytes()
    try:
        source = _object(json.loads(raw), "scenario")
    except (ValueError, UnicodeError) as exc:
        raise PisasError(f"Invalid scenario.json: {exc}") from exc
    family = _component(task_family or source_path.parent.parent.name, "task family")
    name = _component(source_path.parent.name, "scenario folder")
    cast = source.get("cast")
    if not isinstance(cast, dict) or len(cast) < 2:
        raise PisasError("cast must contain at least two named participants")
    participants = []
    names_to_ids = {}
    for index, member in enumerate(cast.values(), 1):
        member = _object(member, "cast member")
        human_name = _text(member.get("name"), "cast member name")
        if human_name in names_to_ids:
            raise PisasError(f"Duplicate participant name: {human_name}")
        principal = {
            "id": f"agent-{index:03d}", "name": human_name,
            "role": _text(member.get("role"), "cast member role"),
            "team": _text(member.get("team", ""), "cast member team", allow_empty=True),
        }
        participants.append(principal)
        names_to_ids[human_name] = principal["id"]
    task = _object(source.get("task"), "task")
    roles = _object(task.get("participants"), "task.participants")
    executor = _object(roles.get(task.get("executor_role")), "task executor")
    executor_name = _text(executor.get("name"), "task executor name")
    if executor_name not in names_to_ids:
        raise PisasError("The executor must be a named cast member")
    timeline = source.get("timeline")
    if not isinstance(timeline, list) or any(not isinstance(item, dict) for item in timeline):
        raise PisasError("timeline must be a list of artifact objects")
    task_text, normalizations = _task_text(source, executor_name, allow_task_fallback=allow_task_fallback)
    artifacts = {principal["id"]: [] for principal in participants}
    for index, entry in enumerate(timeline, 1):
        visible = entry.get("visible_to")
        if not isinstance(visible, list) or any(not isinstance(person, str) for person in visible):
            raise PisasError("Every timeline entry needs a list of visible_to names")
        unknown = set(visible) - names_to_ids.keys()
        if unknown:
            raise PisasError(f"Unknown initial artifact recipients: {', '.join(sorted(unknown))}")
        if entry.get("type") == "task":
            continue
        artifact = {
            "artifact_id": f"artifact-{index:05d}",
            "type": _text(entry.get("type"), "artifact type"),
            "text": _text(entry.get("content"), "artifact content"),
            "author": _text(entry.get("author", ""), "artifact author", allow_empty=True),
            "timestamp": _text(entry.get("timestamp", ""), "artifact timestamp", allow_empty=True),
        }
        for human_name in dict.fromkeys(visible):
            artifacts[names_to_ids[human_name]].append(dict(artifact))
    org = _object(source.get("org", {}), "org")
    source_relations = org.get("relations", [])
    if not isinstance(source_relations, list):
        raise PisasError("org.relations must be a list")
    relations = []
    for relation in source_relations:
        relation = _object(relation, "organizational relation")
        relations.append({key: _text(relation.get(key), f"relation {key}")
                          for key in ("from", "type", "to")})
    recipient = task.get("recipient") or executor_name
    return {
        "schema_version": 1, "benchmark": "PiSAs",
        "case_id": f"{family}__{name}", "task_family": family, "scenario_name": name,
        "task": task_text, "executor_id": names_to_ids[executor_name],
        "recipient": _text(recipient, "task recipient"),
        "participants": participants, "relations": relations, "artifacts": artifacts,
        "normalizations": normalizations,
        "source": {"scenario_sha256": _sha256(raw), "relative_path": f"{family}/{name}/scenario.json"},
    }
