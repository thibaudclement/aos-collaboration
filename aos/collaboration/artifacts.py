# Owner-local, versioned text evidence with a deterministic lexical retriever
# Only explicitly registered files are indexed. This module does not discover files,
# infer permissions, or read PiSAs evaluator annotations. Evidence dictionaries are
# private service data; the release module constructs the smaller public view.

from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import TYPE_CHECKING, Iterable, Iterator
import uuid

if TYPE_CHECKING:
    from .policy import RequestContext

MAX_FILE_BYTES = 8 * 1024 * 1024
SCHEMA = """
CREATE TABLE IF NOT EXISTS identity (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    owner_id TEXT NOT NULL, workspace TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY, source_path TEXT UNIQUE NOT NULL,
    version TEXT NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL
);
"""


# Signal invalid, missing, or changed owner-local evidence.
class ArtifactError(ValueError):
    pass


# Hash a JSON-compatible value without depending on dictionary insertion order.
def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Preserve the exact source version and character offsets of one evidence unit.
@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    artifact_id: str
    owner_id: str
    version: str
    source_path: str
    start: int
    end: int
    text: str

    # Export private evidence for the owner service or the privileged audit.
    def as_dict(self) -> dict:
        return asdict(self)

# Bind retrieved evidence to one runtime-established request context.
@dataclass(frozen=True)
class EvidenceBundle:
    context_hash: str
    owner_id: str
    units: tuple[EvidenceUnit, ...]

    # Freeze an owner-local retrieval result for a particular request.
    @classmethod
    def create(cls, context: RequestContext, units: Iterable[EvidenceUnit]) -> EvidenceBundle:
        rows = tuple(units)
        if any(row.owner_id != context.owner_id for row in rows):
            raise ArtifactError("Evidence belongs to a different owner.")
        if len({row.evidence_id for row in rows}) != len(rows):
            raise ArtifactError("An evidence bundle cannot contain duplicate IDs.")
        return cls(context.digest(), context.owner_id, rows)

    # Resolve an ID only within the evidence retrieved for this request.
    def get(self, evidence_id: str) -> EvidenceUnit:
        for row in self.units:
            if row.evidence_id == evidence_id:
                return row
        raise ArtifactError("Evidence ID is outside the request's retrieval result.")

    # Export the frozen private retrieval record for audit and review.
    def as_dict(self) -> dict:
        return {"context_hash": self.context_hash, "owner_id": self.owner_id,
                "units": [row.as_dict() for row in self.units]}

# Tokenize case-insensitively for the intentionally simple lexical baseline.
def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())

# Divide the complete source into bounded, nonoverlapping character spans.
def _spans(text: str, size: int) -> list[tuple[int, int]]:
    spans = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary < 0:
                boundary = text.rfind(" ", start + size // 2, end)
            if boundary >= 0:
                end = boundary + 1
        spans.append((start, end))
        start = end
    return spans

# Index only a configured owner's explicitly registered UTF-8 artifact files.
class ArtifactIndex:
    # Bind the index permanently to one workspace and keep it outside agent files.
    def __init__(self, db_path: str | Path, workspace: str | Path, owner_id: str,
                 *, chunk_chars: int = 1600):
        self.workspace = Path(workspace).resolve()
        self.db_path = Path(db_path).resolve()
        self.owner_id = owner_id
        self.chunk_chars = chunk_chars
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ArtifactError("The index requires a nonempty owner ID.")
        if not self.workspace.is_dir():
            raise ArtifactError("The owner workspace must already exist.")
        if self.db_path.is_relative_to(self.workspace):
            raise ArtifactError("The evidence index must be outside the agent workspace.")
        if type(chunk_chars) is not int or not 64 <= chunk_chars <= 16_000:
            raise ArtifactError("chunk_chars must be an integer between 64 and 16000.")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            identity = conn.execute("SELECT owner_id, workspace FROM identity").fetchone()
            if identity is None:
                conn.execute("INSERT INTO identity VALUES (1, ?, ?)",
                             (owner_id, str(self.workspace)))
            elif (identity["owner_id"], identity["workspace"]) != (owner_id, str(self.workspace)):
                raise ArtifactError("This index is already bound to another owner or workspace.")

    # Open a short-lived connection so callers never share SQLite connections.
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # Confine source paths to the configured owner workspace, including symlinks.
    def _path(self, source_path: str) -> Path:
        if not isinstance(source_path, str) or not source_path or Path(source_path).is_absolute():
            raise ArtifactError("An artifact path must be workspace-relative.")
        path = (self.workspace / source_path).resolve()
        if not path.is_relative_to(self.workspace):
            raise ArtifactError("Artifact paths cannot escape the owner workspace.")
        return path

    # Read the whole source without silently truncating bytes or replacing characters.
    def _source(self, source_path: str) -> tuple[str, str]:
        path = self._path(source_path)
        try:
            with path.open("rb") as handle:
                raw = handle.read(MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise ArtifactError("An indexed artifact is unavailable.") from exc
        if len(raw) > MAX_FILE_BYTES:
            raise ArtifactError("An artifact exceeds the 8 MiB indexing limit.")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactError("Artifacts must be UTF-8 text before indexing.") from exc
        if "\0" in text:
            raise ArtifactError("Binary artifacts must be converted to text before indexing.")
        return text, hashlib.sha256(raw).hexdigest()

    # Register a whole file and invalidate old unit IDs when its content changes.
    def register_file(self, relative_path: str, artifact_id: str, *, kind: str = "text",
                      source: str = "user") -> list[EvidenceUnit]:
        if not isinstance(artifact_id, str) or not artifact_id.strip() or len(artifact_id) > 200:
            raise ArtifactError("An artifact ID must contain 1 to 200 characters.")
        if not isinstance(kind, str) or not isinstance(source, str):
            raise ArtifactError("Artifact kind and source must be strings.")
        path = self._path(relative_path).relative_to(self.workspace).as_posix()
        text, version = self._source(path)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
            other = conn.execute("SELECT artifact_id FROM artifacts WHERE source_path = ?", (path,)).fetchone()
            if other is not None and other["artifact_id"] != artifact_id:
                raise ArtifactError("A source file is already registered under another artifact ID.")
            unchanged = old is not None and old["version"] == version and old["source_path"] == path
            if not unchanged:
                conn.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))
                conn.execute("INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)",
                             (artifact_id, path, version, kind, source))
                for start, end in _spans(text, self.chunk_chars):
                    conn.execute("INSERT INTO evidence VALUES (?, ?, ?, ?, ?)",
                                 ("ev-" + uuid.uuid4().hex, artifact_id, start, end, text[start:end]))
            rows = conn.execute(self._select() + " WHERE a.artifact_id = ? ORDER BY e.start",
                                (artifact_id,)).fetchall()
        return [self._unit(row) for row in rows]

    # Select evidence together with its current registered source version.
    @staticmethod
    def _select() -> str:
        return ("SELECT e.evidence_id, e.artifact_id, e.start, e.end, e.text, "
                "a.source_path, a.version FROM evidence e JOIN artifacts a "
                "ON a.artifact_id = e.artifact_id")

    # Convert an internal SQLite row into an immutable owner-bound evidence unit.
    def _unit(self, row: sqlite3.Row) -> EvidenceUnit:
        return EvidenceUnit(owner_id=self.owner_id, **dict(row))

    # Refuse stale or modified source content before it reaches retrieval or release.
    def _validate_sources(self, units: Iterable[EvidenceUnit]) -> None:
        checked: dict[str, tuple[str, str]] = {}
        for unit in units:
            if unit.source_path not in checked:
                checked[unit.source_path] = self._source(unit.source_path)
            text, version = checked[unit.source_path]
            if version != unit.version or text[unit.start:unit.end] != unit.text:
                raise ArtifactError("Artifact content changed; reindex before using its evidence.")

    # Resolve one registered unit and check its source again at the point of use.
    def read(self, evidence_id: str) -> EvidenceUnit:
        with self._connect() as conn:
            row = conn.execute(self._select() + " WHERE e.evidence_id = ?", (evidence_id,)).fetchone()
        if row is None:
            raise ArtifactError("Evidence ID is not in this owner's current index.")
        unit = self._unit(row)
        self._validate_sources((unit,))
        return unit

    # Return a lexical top-k result, or the full owner corpus when limit is None.
    def search(self, query: str, limit: int | None = 8) -> list[EvidenceUnit]:
        if not isinstance(query, str) or len(query) > 16_000:
            raise ArtifactError("A retrieval query must be a string of at most 16000 characters.")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 10_000):
            raise ArtifactError("Retrieval limit must be None or an integer from 1 to 10000.")
        with self._connect() as conn:
            rows = conn.execute(self._select() + " ORDER BY a.artifact_id, e.start").fetchall()
        units = [self._unit(row) for row in rows]
        self._validate_sources(units)
        if limit is None:
            return units
        wanted = set(_tokens(query))
        if not wanted:
            return []
        counts = [Counter(_tokens(unit.text)) for unit in units]
        frequencies = {term: sum(term in count for count in counts) for term in wanted}
        ranked = []
        for unit, count in zip(units, counts):
            length = max(sum(count.values()), 1)
            score = sum((1 + math.log(count[term])) * math.log(1 + len(units) / (1 + frequencies[term]))
                        for term in wanted if count[term]) / math.sqrt(length)
            if score > 0:
                ranked.append((score, unit))
        ranked.sort(key=lambda item: (-item[0], item[1].artifact_id, item[1].start))
        return [unit for _, unit in ranked[:limit]]

    # Remove obsolete index entries when the trusted service manifest narrows its source list.
    def retain_artifacts(self, artifact_ids: Iterable[str]) -> None:
        if isinstance(artifact_ids, (str, bytes)):
            raise ArtifactError("An artifact manifest must be an iterable of IDs.")
        ids = tuple(artifact_ids)
        if any(not isinstance(item, str) or not item.strip() for item in ids) or len(ids) != len(set(ids)):
            raise ArtifactError("An artifact manifest requires unique, nonempty IDs.")
        wanted = set(ids)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = {row["artifact_id"] for row in conn.execute("SELECT artifact_id FROM artifacts")}
            if wanted - existing:
                raise ArtifactError("The manifest names an unregistered artifact.")
            conn.executemany("DELETE FROM artifacts WHERE artifact_id = ?",
                             ((artifact_id,) for artifact_id in existing - wanted))

    # Export private registration metadata without reading any unregistered file.
    def snapshot(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT a.*, COUNT(e.evidence_id) AS unit_count FROM artifacts a "
                                "LEFT JOIN evidence e ON e.artifact_id = a.artifact_id "
                                "GROUP BY a.artifact_id ORDER BY a.artifact_id").fetchall()
        return [dict(row) for row in rows]