# Small trusted-runtime helpers (none exposed as an agent tool).

from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


# Reject identifiers that could be interpreted as paths or control characters.
def identifier(value: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value) or value in {".", ".."}:
        raise ValueError("Invalid identifier")
    return value


# Serialize protocol objects consistently for hashing and persistence.
def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# Hash canonical protocol objects without exposing their contents.
def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


# Return an unambiguous UTC timestamp for execution events.
def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# Read a JSON object from a trusted runtime file.
def read_json(path: str | Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


# Atomically write a private runtime record without partially written JSON.
def write_json(path: str | Path, value, *, private: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600 if private else 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Refuse evaluator or control files placed inside an agent's AOS workspace.
def outside_workspace(path: Path, workspace: Path) -> Path:
    path, workspace = path.resolve(), workspace.resolve()
    if path == workspace or path.is_relative_to(workspace):
        raise ValueError("Control and evaluation records must be outside the agent workspace")
    return path

