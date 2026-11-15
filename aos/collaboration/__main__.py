# Run one local owner service for a collaboration worker.

from __future__ import annotations
import argparse
import signal
import sys
import threading
from pathlib import Path
from .service import ArtifactService


# Run the owner server with graceful keyboard and process-manager shutdown.
def serve(config: Path) -> int:
    service = ArtifactService(config)
    stopping = threading.Event()

    # Convert stop signals into a bounded service shutdown request.
    def stop_signal(signum, frame):
        stopping.set()

    previous = {sig: signal.signal(sig, stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        service.start()
        print(f"{service.owner_id} ready at {service.url}", flush=True)
        stopping.wait()
    finally:
        service.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


# Start only the owner service selected by its private configuration.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aos.collaboration")
    sub = parser.add_subparsers(dest="command", required=True)
    owner = sub.add_parser("serve", help="Run one owner service until interrupted")
    owner.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        return serve(args.config)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


# Execute the service worker entry point.
if __name__ == "__main__":
    raise SystemExit(main())
