# Start the populated local collaboration demo with optional model credentials.

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
from dotenv import load_dotenv
from .bootstrap import seed_bundle
from .config import load_configuration
from .server import CollaborationServer
from .store import DemoStore, StoreError


# Expose only the local settings needed to run or inspect the demo.
def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="aos-collaboration", description="Replay saved collaborations and run PiSAs scenarios locally.")
    result.add_argument("--port", type=int, default=8770, help="Local HTTP port (default: 8770)")
    result.add_argument("--workspace", type=Path, default=Path.cwd() / ".aos-collaboration",
                        help="Writable conversation workspace (default: ./.aos-collaboration)")
    result.add_argument("--env-file", type=Path, default=Path.cwd() / ".env", help="Optional environment file (default: ./.env)")
    result.add_argument("--model", help="Actor model; otherwise use AOS_MODEL from .env")
    result.add_argument("--offline", action="store_true", help="Disable new model calls even when an API key is configured")
    return result


# Populate the workspace and serve replay without making a startup model call.
def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        print("Choose a port between 0 and 65535.", file=sys.stderr)
        return 2
    load_dotenv(args.env_file.expanduser(), override=False)
    config = load_configuration(args.model)
    allow_live = config["live_enabled"] and not args.offline
    if allow_live and shutil.which("git") is None:
        print("Git is required for new collaborations. Install Git or use --offline for replay.", file=sys.stderr)
        return 2
    store, server = None, None
    try:
        reason = ("Restart without --offline to enable new collaborations." if args.offline
                  else config["disabled_reason"])
        store = DemoStore(args.workspace, allow_live=allow_live, model=config["model"], disabled_reason=reason)
        bundle = seed_bundle(store)
        server = CollaborationServer(store, port=args.port)
        print(f"AOS Collaboration: {server.url}/collaboration", flush=True)
        print(f"Loaded {bundle['scenarios']} PiSAs scenarios and {bundle['replays']} saved conversations.", flush=True)
        if allow_live:
            print(f"New collaborations use {config['model']}. Each start uses your API credits; no judge is invoked.", flush=True)
        elif args.offline:
            print("Replay-only mode: new model calls are disabled by --offline.", flush=True)
        else:
            print("Replay is available without model calls. Set API_KEY in .env and restart to enable new collaborations.", flush=True)
        print("Press Ctrl+C to stop the server. Accepted collaborations continue in their local worker.", flush=True)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopped the local collaboration server.", flush=True)
    except OSError as exc:
        message = (f"Port {args.port} is already in use; try --port {args.port + 1}."
                   if getattr(exc, "errno", None) in {48, 98, 10048}
                   else f"Cannot open the local demo workspace or server ({type(exc).__name__}).")
        print(message, file=sys.stderr)
        return 2
    except (StoreError, ValueError) as exc:
        print(f"Cannot start the demo: {exc}", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.server_close()
        if store is not None:
            store.close()
    return 0


# Support the same launcher through the Python module entry point.
if __name__ == "__main__":
    raise SystemExit(main())
