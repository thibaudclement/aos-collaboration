# AOS Collaboration

A focused demo of collaboration between agents, built on AOS’s workspace, task, and model-client infrastructure: request and review evidence from separate workspaces, then record a shared decision. It includes 265 PiSAs scenarios and 109 saved conversations, with `governed-v1`, `observed-v2`, and `integrated-v3` available for new collaborations.

## Run locally

Requires Python 3.10+ and Git on macOS or Linux (Windows: use WSL).

```bash
git clone https://github.com/thibaudclement/aos-collaboration.git
cd aos-collaboration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp -n .env.example .env
```

For new collaborations, set `API_KEY` in `.env`. The defaults use OpenRouter, so use an OpenRouter key with those settings. To use another OpenAI-compatible provider, also set `BASE_URL` and `AOS_MODEL` to its endpoint and model identifier. Leave `API_KEY` blank to browse and replay saved runs without model calls. Each new collaboration uses your API credits.

```bash
aos-collaboration
```

Open **http://127.0.0.1:8770/collaboration**. The catalog and saved runs load automatically. Search the sidebar for `External baseline`, `Calibration`, a task, or a system; select a scenario and decision behavior to start a new collaboration. Saved evaluations are displayed when available; replay and live collaboration never invoke a judge.

Local conversations persist in `.aos-collaboration/`, which is excluded from Git. Use `aos-collaboration --offline` to force replay-only mode, or `aos-collaboration --port 8771` if the default port is occupied. Press Ctrl+C to stop the server; an already-started collaboration continues in its local worker.

