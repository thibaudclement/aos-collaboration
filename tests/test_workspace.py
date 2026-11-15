import json
import pytest
from aos.workspace import STANDARD_DIRS, Workspace, WorkspaceError, file_type


def test_create_makes_folders_metadata_and_initial_commit(tmp_path):
    ws = Workspace.create(tmp_path / "ws", goal="find patients")
    for d in STANDARD_DIRS:
        assert (ws.root / d).is_dir()
    assert (ws.root / "workspace.db").is_file()
    assert ws.goal == "find patients" and ws.files() == {}
    assert (ws.root / ".git").is_dir()
    assert [e["message"] for e in ws.history()] == ["Initialize workspace"]
    assert not ws.has_changes()


def test_create_is_idempotent_and_open_requires_workspace(tmp_path):
    ws = Workspace.create(tmp_path / "ws", goal="g")
    again = Workspace.create(tmp_path / "ws")
    assert again.goal == "g"
    assert len(ws.history()) == 1
    with pytest.raises(WorkspaceError):
        Workspace.open(tmp_path / "nope")


def test_relpath_rejects_paths_outside_workspace(ws, tmp_path):
    assert ws.relpath("inputs/a.csv") == "inputs/a.csv"
    assert ws.relpath(ws.root / "generated" / "x.md") == "generated/x.md"
    with pytest.raises(WorkspaceError):
        ws.relpath("../elsewhere.txt")
    with pytest.raises(WorkspaceError):
        ws.relpath(tmp_path / "other")


def test_register_and_describe(ws):
    (ws.root / "inputs" / "patients.csv").write_text("id,age\n1,40\n")
    entry = ws.register("inputs/patients.csv", source="user", reason="patient records",
                        summary="two columns", task="task-001", inputs=[], tags=["input"])
    assert entry["type"] == "csv" and entry["size"] == 12 and entry["agent"] == "user"
    stored = ws.describe(ws.root / "inputs/patients.csv")
    assert stored["reason"] == "patient records" and stored["tags"] == ["input"]
    assert stored["created"] == entry["created"]
    # re-registering keeps the creation time and updates the rest
    updated = ws.register("inputs/patients.csv", source="sliders", reason="re-derived", inputs=["inputs/patients.csv"])
    assert updated["created"] == entry["created"] and updated["source"] == "sliders"
    assert updated["inputs"] == ["inputs/patients.csv"]
    assert ws.describe("generated/missing.md") is None


def test_snapshot_changed_since_and_unregistered(ws):
    (ws.root / "inputs" / "a.txt").write_text("a")
    before = ws.snapshot()
    assert set(before) == {"inputs/a.txt"}  # .gitkeep, metadata.json and tasks/ are not work products
    (ws.root / "generated" / "reports").mkdir()
    (ws.root / "generated" / "reports" / "r.md").write_text("# report")
    (ws.root / "inputs" / "a.txt").write_text("changed content")
    (ws.root / "tasks" / "task-001").mkdir()
    (ws.root / "tasks" / "task-001" / "task.json").write_text("{}")
    (ws.root / "code" / "__pycache__").mkdir()
    (ws.root / "code" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (ws.root / "generated" / ".cache").mkdir()
    (ws.root / "generated" / ".cache" / "chunks.parquet").write_bytes(b"\0")  # caches are outputs too
    assert ws.changed_since(before) == ["generated/.cache/chunks.parquet", "generated/reports/r.md", "inputs/a.txt"]
    ws.register("inputs/a.txt", source="user", reason="input")
    ws.register("generated/.cache/chunks.parquet", source="sliders", reason="store")
    assert ws.unregistered() == ["generated/reports/r.md"]
    assert "generated/.cache/ — [sliders] 1 cache/store files (hidden)" in ws.tree()
    assert "chunks.parquet — " not in ws.tree()
    assert "generated/reports/r.md — (unregistered)" in ws.tree()
    assert "inputs/a.txt — [user] input" in ws.tree()


def test_prune_metadata_drops_deleted_files(ws):
    (ws.root / "generated" / "tmp.txt").write_text("x")
    ws.register("generated/tmp.txt", source="coding_agent", reason="scratch")
    (ws.root / "generated" / "tmp.txt").unlink()
    assert ws.prune_metadata() == ["generated/tmp.txt"]
    assert ws.files() == {}


def test_checkpoint_commits_only_when_something_changed(ws):
    assert ws.checkpoint("nothing") is None
    (ws.root / "generated" / "out.md").write_text("hi")
    sha = ws.checkpoint("add output")
    assert sha == ws.head() and len(sha) == 40
    assert ws.history()[0]["message"] == "add output"
    assert not ws.has_changes()


def test_rollback_restores_files_and_metadata_but_keeps_task_records(ws):
    (ws.root / "generated" / "v1.md").write_text("version 1")
    ws.register("generated/v1.md", source="storm", reason="first")
    good = ws.checkpoint("step 1")

    (ws.root / "generated" / "v1.md").write_text("broken")
    (ws.root / "generated" / "extra.md").write_text("should disappear")
    ws.register("generated/extra.md", source="coding_agent", reason="bad step")
    (ws.root / "tasks" / "task-001").mkdir(parents=True)
    (ws.root / "tasks" / "task-001" / "task.json").write_text('{"steps": 2}')
    ws.checkpoint("step 2 (bad)")

    sha = ws.rollback(good)
    assert (ws.root / "generated" / "v1.md").read_text() == "version 1"
    assert not (ws.root / "generated" / "extra.md").exists()
    assert set(ws.files()) == {"generated/v1.md"}
    assert (ws.root / "tasks" / "task-001" / "task.json").read_text() == '{"steps": 2}'
    assert ws.head() == sha
    messages = [e["message"] for e in ws.history()]
    assert messages[0].startswith("Rollback workspace to") and "step 2 (bad)" in messages
    assert not ws.has_changes()


def test_modified_and_mark_edited(ws):
    (ws.root / "generated").mkdir(exist_ok=True)
    (ws.root / "generated" / "out.md").write_text("v1")
    ws.register("generated/out.md", source="storm", reason="research", summary="v1", task="task-001", step=1, inputs=["inputs/a.txt"])
    ws.checkpoint("step")
    assert ws.modified() == []
    (ws.root / "generated" / "out.md").write_text("v2 by hand")
    assert ws.modified() == ["generated/out.md"]
    (ws.root / "inputs" / "a.txt").unlink() if (ws.root / "inputs" / "a.txt").exists() else None
    assert ws.modified() == ["generated/out.md"]  # a deleted file is not an edit
    e = ws.mark_edited("generated/out.md", "v2 summary")
    assert e["source"] == "mixed" and e["agent"] == "storm" and e["task"] == "task-001" and e["step"] == 1
    assert e["tags"] == ["edited-by-user"] and e["summary"] == "v2 summary" and e["inputs"] == ["inputs/a.txt"]
    assert ws.mark_edited("generated/out.md")["tags"] == ["edited-by-user"]  # idempotent, summary kept
    with pytest.raises(WorkspaceError):
        ws.mark_edited("generated/never.md")


def test_settings_round_trip(ws):
    assert ws.setting("budget_usd") is None and ws.setting("budget_usd", 5) == 5
    ws.set_setting("budget_usd", 12.5)
    assert ws.setting("budget_usd") == 12.5 and ws.goal == "test goal"
    ws.set_setting("budget_usd", None)
    assert ws.setting("budget_usd", "unset") is None


def test_file_type():
    assert file_type("a/b.CSV") == "csv"
    assert file_type("report.md") == "markdown"
    assert file_type("data.db") == "sqlite"
    assert file_type("weird.xyz") == "xyz"
    assert file_type("Makefile") == "file"


def test_tree_collapses_large_directories(ws):
    (ws.root / "generated" / "run" / "tasks").mkdir(parents=True)
    for i in range(30):
        (ws.root / "generated" / "run" / "tasks" / f"{i:04d}_prompt.txt").write_text("x")
    (ws.root / "generated" / "report.md").write_text("r")
    ws.register("generated/report.md", source="grill", reason="research", summary="the report")
    tree = ws.tree()
    assert "generated/report.md — [grill] the report" in tree
    assert tree.count("generated/run/tasks/") == 6  # five examples plus the collapsed line
    assert "generated/run/tasks/... — 25 more files like these" in tree


def test_query_is_read_only_and_joins_inputs(ws):
    (ws.root / "inputs" / "a.csv").write_text("x")
    (ws.root / "generated" / "r.md").write_text("y")
    ws.register("inputs/a.csv", source="user", reason="input", task="task-001")
    ws.register("generated/r.md", source="storm", reason="research", summary="the report",
                task="task-001", step=2, inputs=["inputs/a.csv"], tags=["report"])
    rows = ws.query("SELECT path, summary FROM files WHERE source = ? ORDER BY path", ("storm",))
    assert rows == [{"path": "generated/r.md", "summary": "the report"}]
    derived = ws.query("SELECT f.path FROM files f JOIN file_inputs i ON i.path = f.path WHERE i.input = 'inputs/a.csv'")
    assert [r["path"] for r in derived] == ["generated/r.md"]
    assert ws.query("SELECT tags FROM files WHERE path = 'generated/r.md'")[0]["tags"] == '["report"]'
    with pytest.raises(WorkspaceError, match="only SELECT"):
        ws.query("DELETE FROM files")
    with pytest.raises(WorkspaceError, match="query failed"):
        ws.query("SELECT nope FROM files")
    assert ws.files()["generated/r.md"]["inputs"] == ["inputs/a.csv"]


def test_legacy_metadata_json_is_migrated(tmp_path):
    ws = Workspace.create(tmp_path / "ws", goal="old")
    ws.db_path.unlink()
    (ws.root / "metadata.json").write_text(json.dumps({"created": "2026-01-01T00:00:00+00:00", "goal": "legacy goal", "files": {
        "inputs/p.csv": {"path": "inputs/p.csv", "type": "csv", "source": "user", "reason": "input", "summary": "patients",
                         "task": "task-001", "step": None, "agent": "user", "inputs": [], "tags": ["input"],
                         "created": "2026-01-01T00:00:00+00:00", "updated": "2026-01-01T00:00:00+00:00", "size": 3}}}))
    reopened = Workspace.open(ws.root)
    assert reopened.goal == "legacy goal"
    entry = reopened.describe("inputs/p.csv")
    assert entry["summary"] == "patients" and entry["tags"] == ["input"] and entry["created"] == "2026-01-01T00:00:00+00:00"
    assert not (ws.root / "metadata.json").exists()
