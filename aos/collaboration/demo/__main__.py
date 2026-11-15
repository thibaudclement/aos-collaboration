# Manage local replay records and launch the AOS collaboration interface.

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .config import DEFAULT_MODEL
from .store import DemoStore, StoreError

# Define explicit operator actions (no command scans reserved cases or invokes an official judge).
def parser():
    root = argparse.ArgumentParser(prog="python -m aos.collaboration.demo", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (("init", "Create a private local demo store with an offline fixture"),
                            ("add-demo", "Register the zero-cost authored fixture"),
                            ("status", "List registered cases and conversations without provider calls")):
        item = commands.add_parser(name, help=help_text)
        item.add_argument("root", type=Path)
    replay = commands.add_parser("add-replay", help="Import a saved AOS run as a faithful requester view")
    replay.add_argument("root", type=Path)
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--title")
    replay.add_argument("--pipeline", type=Path)
    replay.add_argument("--evaluation", type=Path)
    native = commands.add_parser("add-native-replay", help="Import saved native PiSAs messages without an AOS release audit")
    native.add_argument("root", type=Path)
    native.add_argument("--pipeline", type=Path, required=True)
    native.add_argument("--case", required=True, help="Explicit task_family/scenario_NN")
    native.add_argument("--run-index", type=int, default=0)
    native.add_argument("--title")
    native.add_argument("--evaluation", type=Path)
    case = commands.add_parser("add-case", help="Register one PiSAs scenario without reading its labels")
    case.add_argument("root", type=Path)
    case.add_argument("--scenario", type=Path, required=True)
    web = commands.add_parser("serve", help="Serve replay and controlled live collaboration on loopback")
    web.add_argument("root", type=Path)
    web.add_argument("--port", type=int, default=8770)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--enable-live", action="store_true", help="Enable explicitly started model-backed cases; never launches a judge")
    web.add_argument("--model", default=DEFAULT_MODEL)
    web.add_argument("--top-k", type=int, default=8)
    worker = commands.add_parser("execute", help=argparse.SUPPRESS)
    worker.add_argument("root", type=Path)
    worker.add_argument("run_id")
    return root

# Dispatch one operator command while keeping failed starts and failures visible.
def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "serve":
        from .server import serve
        serve(args.root, host=args.host, port=args.port, allow_live=args.enable_live,
              model=args.model, top_k=args.top_k)
        return 0
    store = DemoStore(args.root)
    try:
        if args.command == "add-case":
            result = store.add_case(args.scenario)
        elif args.command == "add-replay":
            result = store.add_replay(args.run, title=args.title, pipeline=args.pipeline, evaluation=args.evaluation)
        elif args.command == "add-native-replay":
            result = store.add_native_replay(args.pipeline, case=args.case, run_index=args.run_index,
                                             title=args.title, evaluation=args.evaluation)
        elif args.command == "add-demo":
            result = store.add_demo()
        elif args.command == "execute":
            result = store.execute(args.run_id)
        else:
            result = {"root": str(store.root), "catalog": store.catalog(), **store.runs(), "provider_calls": 0}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") != "failed" else 2
    except (OSError, ValueError, KeyError) as exc:
        message = str(exc) if isinstance(exc, (StoreError, ValueError)) else type(exc).__name__
        print("Collaboration demo stopped: " + message, file=sys.stderr)
        return 2
    finally:
        store.close()

# Support the package CLI without changing existing artifact-request commands.
if __name__ == "__main__":
    raise SystemExit(main())
