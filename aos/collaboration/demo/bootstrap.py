# Load the bundled catalogue and immutable recorded conversations into a local workspace.

from __future__ import annotations
import copy
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from ..pisas import load_scenario
from .store import RUN_ID, StoreError, read_bounded, write_private

BUNDLE_SCHEMA = "aos-collaboration-bundle-v1"
SEED_SCHEMA = "aos-collaboration-replay-seed-v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


# Locate bundled data in either an editable checkout or an installed wheel.
def bundled_data() -> Path:
    package = Path(__file__).resolve().parents[2]
    for candidate in (package / "data", package.parent / "data"):
        if (candidate / "manifest.json").is_file():
            return candidate
    raise StoreError("Bundled demo data is missing; extract the complete archive or reinstall the package.")


# Verify a manifest-owned file without allowing a path outside the bundle.
def _verified_file(root: Path, item: dict, prefix: str) -> tuple[Path, bytes]:
    name, expected = item.get("path"), item.get("sha256")
    if not isinstance(name, str) or not isinstance(expected, str) or not SHA256.fullmatch(expected):
        raise StoreError("Invalid bundled file entry.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != prefix:
        raise StoreError("Invalid bundled file path.")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise StoreError("A bundled data file is missing.")
    if path.stat().st_size > 128 * 1024 * 1024:
        raise StoreError("A bundled data file exceeds the size limit.")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise StoreError("Bundled data failed its integrity check; use an unchanged copy of the archive.")
    return path, raw


# Copy scenario bytes atomically while preserving their recorded source hash.
def _copy_scenario(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise StoreError("A workspace scenario differs from the bundled revision; choose a new --workspace.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".scenario-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Validate all seed inputs before publishing their catalogue and replay records.
def seed_bundle(store, root: Path | None = None) -> dict:
    root = (root or bundled_data()).resolve()
    manifest, manifest_sha256 = read_bounded(root / "manifest.json")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise StoreError("Unsupported bundled catalogue.")
    scenarios, replays = manifest.get("scenarios"), manifest.get("replays")
    if not isinstance(scenarios, list) or not isinstance(replays, list):
        raise StoreError("The bundled catalogue is incomplete.")
    cases, seeds, seen = [], [], set()
    for item in scenarios:
        path, raw = _verified_file(root, item, "scenarios")
        projection = load_scenario(path)
        case_id = projection["case_id"]
        if item.get("case_id") != case_id or case_id in seen:
            raise StoreError("Invalid or duplicate bundled scenario identity.")
        seen.add(case_id)
        destination = store.root / "catalog" / projection["task_family"] / projection["scenario_name"] / "scenario.json"
        cases.append(({"case_id": case_id, "title": f"{projection['task_family']} / {projection['scenario_name']}",
                       "task": projection["task"], "family": projection["task_family"],
                       "scenario": projection["scenario_name"], "kind": "pisas",
                       "participant_count": len(projection["participants"]),
                       "source_sha256": item["sha256"], "scenario_path": str(destination)}, raw))
    run_ids = set()
    for item in replays:
        path, _ = _verified_file(root, item, "replays")
        seed, _ = read_bounded(path)
        record = seed.get("record", {})
        view = seed.get("view", {})
        run_id = record.get("run_id")
        if (seed.get("schema") != SEED_SCHEMA or not isinstance(run_id, str)
                or not RUN_ID.fullmatch(run_id) or run_id in run_ids or item.get("run_id") != run_id
                or view.get("run_id") != run_id or record.get("case_id") not in seen
                or record.get("mode") not in {"recorded_replay", "native_replay"}
                or record.get("status") not in {"completed", "completed_with_warning", "failed", "interrupted"}
                or seed.get("evaluation", {}).get("status") != record.get("evaluation_status")
                or view.get("source_sha256") != record.get("source_sha256")):
            raise StoreError("Invalid bundled replay identity or status.")
        run_ids.add(run_id)
        seeds.append(seed)
    if manifest.get("counts", {}).get("scenarios") != len(cases) or manifest.get("counts", {}).get("replays") != len(seeds):
        raise StoreError("The bundled catalogue counts do not match its contents.")
    with store._locked() as registry:
        for record, raw in cases:
            existing = registry["cases"].get(record["case_id"])
            if existing and existing["source_sha256"] != record["source_sha256"]:
                raise StoreError("A registered scenario has a different revision; choose a new --workspace.")
        for seed in seeds:
            record = seed["record"]
            existing = registry["runs"].get(record["run_id"])
            if existing and (existing.get("mode") not in {"recorded_replay", "native_replay"}
                             or existing.get("source_sha256") != record["source_sha256"]):
                raise StoreError("A bundled replay identifier conflicts with an existing conversation.")
        for record, raw in cases:
            _copy_scenario(Path(record["scenario_path"]), raw)
            registry["cases"][record["case_id"]] = record
        for seed in seeds:
            record = copy.deepcopy(seed["record"])
            directory = store._directory(record["run_id"])
            for name in ("view", "evaluation", "progress", "provenance"):
                write_private(directory / f"{name}.json", seed.get(name, {}))
            registry["runs"][record["run_id"]] = record
        registry["cases"].pop("availability_demo__scenario_01", None)
        registry["bundle"] = {"schema": BUNDLE_SCHEMA, "manifest_sha256": manifest_sha256,
                              "scenarios": len(cases), "replays": len(seeds)}
    return {"scenarios": len(cases), "replays": len(seeds), "manifest_sha256": manifest_sha256}
