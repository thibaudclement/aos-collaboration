import json
from aos.tasks import Step, Task

def test_new_tasks_get_sequential_ids_and_are_saved(ws):
    t1 = Task.new(ws, "first")
    t2 = Task.new(ws, "second")
    assert (t1.id, t2.id) == ("task-001", "task-002")
    assert json.loads((ws.root / "tasks" / "task-002" / "task.json").read_text())["goal"] == "second"
    assert [t.id for t in Task.list(ws)] == ["task-001", "task-002"]
    assert Task.load(ws, "task-001").goal == "first"


def test_step_records_and_summary_text(ws):
    task = Task.new(ws, "goal")
    record = task.start_step(Step(worker="storm", instructions="research X", reason="need background",
                                  inputs=["inputs/a.txt"], output_dir="generated/reports"))
    assert record["n"] == 1 and record["status"] == "running"
    assert record["log"] == "tasks/task-001/step-01.log"
    task.finish_step(record, status="ok", outputs=["generated/reports/report.md"], summary="wrote report")
    task.finish("done", "all good")
    reloaded = Task.load(ws, task.id)
    step = reloaded.steps[0]
    assert step["status"] == "ok" and step["outputs"] == ["generated/reports/report.md"]
    assert step["finished"] and reloaded.data["status"] == "done"
    text = reloaded.summary_text()
    assert "Step 1 [ok] storm: need background" in text
    assert "inputs: inputs/a.txt" in text and "outputs: generated/reports/report.md" in text
    assert "summary: wrote report" in text


def test_summary_text_without_steps(ws):
    assert Task.new(ws, "g").summary_text() == "(no steps yet)"
