# AOS Collaboration

## GUI demo run

A focused demo of collaboration between agents, built on AOS’s workspace, task, and model-client infrastructure: request and review evidence from separate workspaces, then record a shared decision. It includes 265 PiSAs scenarios and 136 saved conversations, with `governed-v1`, `observed-v2`, and `integrated-v3` available for new collaborations.

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

To launch the demo:

```bash
aos-collaboration
```

Open **http://127.0.0.1:8770/collaboration**. The catalog and saved runs load automatically. Search the sidebar for `External baseline`, `Calibration`, `Reserved wave 3`, a task, or a system; select a scenario and decision behavior to start a new collaboration. The Expected outcome panel shows the bundled PiSAs reference answer, including the full assignment or schedule for allocation tasks (it is not supplied to the agents). Saved evaluations are displayed when available. Replay and live collaboration never invoke a judge.

Local conversations persist in `.aos-collaboration/`, which is excluded from Git. Use `aos-collaboration --offline` to force replay-only mode, or `aos-collaboration --port 8771` if the default port is occupied. Press Ctrl+C to stop the server; an already-started collaboration continues in its local worker.

## CLI experiments replication

A cleaned-up implementation for replicating this project's experiments—rather than the original development codebase. It fixes the original cases, actor models, systems, and response contracts, reuses the AOS actor components and pinned PiSAs evaluator, and produces one consistent export format. Results are nondeterministic. Replication outputs are separate from the demo and do not appear in the GUI automatically.

**IMPORTANT:**

> **Paid execution can take hours and consume substantial API credit.** Wave 3 recorded approximately **$22 in known charges and 6.4 hours of active stage execution**, or about **10 hours elapsed** including interruptions. One request had no recorded price; these figures are historical observations, not a price quote or spending cap. The CLI does not cap spending. Set appropriate limits with the providers before running a batch.

After updating an existing checkout, register the new command:

```bash
python -m pip install -e .
```

All batches require an OpenRouter key in `API_KEY` for paid execution. External baseline additionally requires a Google AI Studio key in `GOOGLE_API_KEY`: its governed actors used Google's endpoint, while its native actors and judges used OpenRouter. The CLI uses the batch's fixed endpoints and models, independently of the demo's `BASE_URL` and `AOS_MODEL`.

| Batch | Cases | Runs | Systems | Actor |
| --- | ---: | ---: | --- | --- |
| `external-baseline` | 3 | 12 | Native PiSAs, governed-v1; two repetitions | Gemini 3.5 Flash Lite |
| `calibration` | 9 | 18 | Native PiSAs, governed-v1 | Gemini 3.5 Flash Lite |
| `wave1` | 9 | 18 | Native PiSAs, governed-v1 | Gemini 3.5 Flash Lite |
| `wave2` | 9 | 27 | Native PiSAs, governed-v1 + RC, integrated-v3 + RC | Gemini 3.5 Flash Lite |
| `wave3` | 9 | 27 | Native PiSAs, governed-v1 + RC, integrated-v3 + RC | Claude Opus 5.5 |

Preview one batch offline, with no API key or provider requests:

```bash
aos-experiments run wave3
```

Replace `wave3` with any batch name above. Only adding `--execute` starts paid actor **and judge** calls:

```bash
aos-experiments run wave3 --execute
```

Outputs are written to `experiments/wave3/`. `plan.json`, `status.json`, `readout.json`, `replay.json`, `replays/`, `analysis.zip`, and per-stage artifacts under `slots/` are excluded from Git. Paid progress is printed in the terminal and saved in `progress.log`. Readouts and analysis archives contain private actor/evaluator evidence. Public replay projections are separate.

Inspect status or rebuild exports offline:

```bash
aos-experiments status wave3
aos-experiments export wave3
```

Notes:
- Repeating an execution command preserves completed stages and continues only from an untouched next stage.
- A failed or interrupted paid stage stops the batch and is not retried automatically.
- Received judge candidates retain the bounded original parsing loops, and exhausted responses become failed evaluations.
- This common failure handling is stricter than some earlier experimental wrappers.
- We recommend keeping the output directory when diagnosing a failure.
- For an independent new replication, use a fresh directory, such as `--out experiments/wave3-repeat1`, on the relevant commands.

Run the offline tests without spending API credit:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```
