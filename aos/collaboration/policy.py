# Runtime-established request contexts and private owner disclosure policies.
# Inferred policies are instructions for a fallible model reviewer, not gold labels
# or a deterministic privacy guarantee. Explicit grants are restricted to declared
# controlled fixtures so a benchmark run cannot silently use an oracle policy.

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable
from .artifacts import EvidenceUnit, canonical_hash


# Reject malformed, misplaced, or changed trusted policy configuration.
class PolicyError(ValueError):
    pass


# Check an exact string field without coercing identities or untrusted values.
def _text(value: object, name: str, maximum: int = 16_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\0" in value:
        raise PolicyError(f"{name} must be a nonempty string of at most {maximum} characters.")
    return value


# Bind the requester, owner, purpose, and question outside model-controlled tools.
@dataclass(frozen=True)
class RequestContext:
    request_id: str
    requester_id: str
    owner_id: str
    purpose: str
    question: str = ""

    # Validate context fields without allowing implicit string conversions.
    def __post_init__(self) -> None:
        for name in ("request_id", "requester_id", "owner_id"):
            _text(getattr(self, name), name, 200)
        _text(self.purpose, "purpose")
        if not isinstance(self.question, str) or len(self.question) > 16_000 or "\0" in self.question:
            raise PolicyError("question must be a string of at most 16000 characters.")

    # Export the context for the owner's private execution and audit record.
    def as_dict(self) -> dict:
        return asdict(self)

    # Bind a grant to the entire question and authorization context, not just an ID.
    def digest(self) -> str:
        return canonical_hash(self.as_dict())


# Describe a synthetic owner's exact approved release view of versioned evidence.
@dataclass(frozen=True)
class ExplicitGrant:
    artifact_id: str
    source_version: str
    source_text: str
    release_text: str
    recipients: tuple[str, ...]
    purposes: tuple[str, ...]

    # Parse a fixture grant with no wildcard recipients or implicit permission widening.
    @classmethod
    def from_dict(cls, value: object) -> ExplicitGrant:
        keys = {"artifact_id", "source_version", "source_text", "release_text", "recipients", "purposes"}
        if not isinstance(value, dict) or set(value) != keys:
            raise PolicyError("Explicit fixture grants require exactly the documented six fields.")
        for name in ("artifact_id", "source_version", "source_text", "release_text"):
            _text(value[name], name)
        if not re.fullmatch(r"[0-9a-f]{64}", value["source_version"]):
            raise PolicyError("A fixture grant must bind an exact SHA-256 source version.")
        parsed = {}
        for name in ("recipients", "purposes"):
            entries = value[name]
            if not isinstance(entries, list) or not entries:
                raise PolicyError(f"{name} must be a nonempty list.")
            entries = tuple(_text(item, name) for item in entries)
            if len(entries) != len(set(entries)) or "*" in entries:
                raise PolicyError("Fixture grant recipients and purposes must be unique exact values.")
            parsed[name] = entries
        return cls(**{name: value[name] for name in keys - {"recipients", "purposes"}}, **parsed)

    # Check one citation and one exact owner-approved release string against the grant.
    def allows(self, context: RequestContext, unit: EvidenceUnit, text: str) -> bool:
        return (context.requester_id in self.recipients and context.purpose in self.purposes
                and unit.owner_id == context.owner_id and unit.artifact_id == self.artifact_id
                and unit.version == self.source_version and self.source_text in unit.text
                and text == self.release_text)


# Freeze one policy revision while extraction and review are in progress.
@dataclass(frozen=True)
class PolicySnapshot:
    owner_id: str
    mode: str
    instructions: str
    version: str
    grants: tuple[ExplicitGrant, ...] = ()

    # Keep private policy configuration available to the reviewer and audit only.
    def as_dict(self) -> dict:
        return {"owner_id": self.owner_id, "mode": self.mode,
                "instructions": self.instructions, "version": self.version,
                "grants": [asdict(grant) for grant in self.grants]}

    # Require every cited source to have an exact grant when running a controlled fixture.
    def allows(self, context: RequestContext, text: str, units: Iterable[EvidenceUnit]) -> bool:
        rows = tuple(units)
        if context.owner_id != self.owner_id or not rows or any(row.owner_id != self.owner_id for row in rows):
            return False
        if self.mode == "inferred":
            return True
        return all(any(grant.allows(context, unit, text) for grant in self.grants) for unit in rows)


# Load owner-controlled policy outside all files exposed to the agent's tools.
class PolicyStore:
    # Reject a policy file inside the agent workspace, including through a symlink.
    def __init__(self, path: str | Path, *, workspace: str | Path, owner_id: str):
        self.path = Path(path).absolute()
        self.workspace = Path(workspace).resolve()
        self.owner_id = _text(owner_id, "owner_id", 200)
        self._check_location()

    # Recheck policy location at every read so a later symlink cannot move it into the workspace.
    def _check_location(self) -> None:
        if self.path.resolve().is_relative_to(self.workspace):
            raise PolicyError("Owner policies must be outside the agent workspace.")

    # Validate and hash the complete trusted configuration before using its instructions.
    def load(self) -> PolicySnapshot:
        self._check_location()
        try:
            with self.path.open("rb") as handle:
                raw = handle.read(256_001)
        except OSError as exc:
            raise PolicyError("The owner policy is unavailable.") from exc
        if len(raw) > 256_000:
            raise PolicyError("The owner policy exceeds the 256000-byte limit.")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise PolicyError("The owner policy must be valid UTF-8 JSON.") from exc
        allowed = {"schema_version", "owner_id", "mode", "instructions", "grants"}
        required = {"schema_version", "owner_id", "mode", "instructions"}
        if not isinstance(value, dict) or not required <= set(value) or set(value) - allowed:
            raise PolicyError("The owner policy has missing or unexpected fields.")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise PolicyError("Unsupported owner policy schema version.")
        if value["owner_id"] != self.owner_id:
            raise PolicyError("The policy belongs to a different owner.")
        if value["mode"] not in {"inferred", "explicit_fixture"}:
            raise PolicyError("Policy mode must be inferred or explicit_fixture.")
        instructions = _text(value["instructions"], "instructions", 32_000)
        grants = value.get("grants", [])
        if not isinstance(grants, list):
            raise PolicyError("Policy grants must be a list.")
        if value["mode"] == "inferred" and grants:
            raise PolicyError("An inferred policy cannot contain explicit or oracle grants.")
        return PolicySnapshot(self.owner_id, value["mode"], instructions, canonical_hash(value),
                              tuple(ExplicitGrant.from_dict(item) for item in grants))

    # Invalidate an approval if the owner changed policy while the request was processed.
    def assert_current(self, snapshot: PolicySnapshot) -> None:
        current = self.load()
        if current != snapshot:
            raise PolicyError("The owner policy changed; a new review is required.")
