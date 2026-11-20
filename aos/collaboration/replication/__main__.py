# Expose fixed replication batches without changing the demo command or workspace.

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from . import batches, runner
from .provider import ProviderError


# Build the small command surface with offline preview as the default behavior.
def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="aos-experiments",
        description="Replicate fixed PiSAs experiments; run is offline unless --execute is supplied.")
    commands = result.add_subparsers(dest="command", required=True)
    names = [row["name"] for row in batches.list_batches()]
    for name, help_text in (("run", "Preview a batch or explicitly execute its paid stages"),
                            ("status", "Inspect saved stage status without model calls"),
                            ("export", "Rebuild readout, analysis, and replay files without model calls")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("batch", choices=names)
        command.add_argument("--out", type=Path, help="Output directory; default: experiments/BATCH")
        if name == "run":
            command.add_argument("--execute", action="store_true",
                help="Authorize paid actor and judge calls; can take hours and substantial API credit")
    return result


# Send readable operational progress to stderr while reserving stdout for a JSON summary.
def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# Present consistent batch-level counts without printing private model or judge content.
def summary(status: dict, output: Path, *, preview: bool = False) -> dict:
    return {"batch": status["phase"], "output": str(output.resolve()), "preview": preview,
            "runs": status["runs"], "stages": status["stages"], "statuses": status["statuses"],
            "model_calls": status["model_calls"], "known_response_cost_usd": status["known_response_cost_usd"],
            "unknown_cost_calls": status["unknown_cost_calls"],
            "unreported_cost_calls": status["unreported_cost_calls"], "stop": status["stop"],
            "next_stage": (status["next"] or {}).get("id"), "gui_imported": False}


# Dispatch only the requested fixed batch and return a nonzero status for incomplete execution.
def main(argv=None) -> int:
    args = parser().parse_args(argv)
    output = args.out or Path("experiments") / args.batch
    try:
        if args.command == "run":
            batch = batches.get_batch(args.batch)
            if not args.execute:
                progress(f"OFFLINE PREVIEW: {batch['title']}; no API key or model calls required.")
                for case in batch["cases"]:
                    progress("  " + case)
                progress("Paid execution requires --execute. Actor and judge calls can take hours; the CLI does not cap spending.")
            result = runner.run(args.batch, output, execute=args.execute, progress=progress)
            value = summary(result["status"], output, preview=result["preview"])
            value["exported_replays"] = result["exports"]["available_runs"]
            value["exports"] = {name: str(output / name) for name in ("status.json", "readout.json", "replay.json", "analysis.zip")}
            print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
            if (result["status"]["statuses"] == {"ok": result["status"]["stages"]}
                    and result["exports"]["available_runs"] != result["exports"]["planned_runs"]):
                progress("Execution finished, but a replay export failed; inspect readout.json and replay.json.")
                return 1
            return 1 if result["status"]["stop"] else 0
        if args.command == "status":
            value = summary(runner.status(args.batch, output), output)
        else:
            value = runner.export(args.batch, output)
        print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        progress("Interrupted. Saved attempts are retained; inspect status before starting any further execution.")
        return 130
    except (ProviderError, runner.ReplicationError, ValueError, OSError) as exc:
        progress(f"Error: {exc}")
        return 1


# Run the same CLI when the package is invoked with python -m.
if __name__ == "__main__":
    raise SystemExit(main())
