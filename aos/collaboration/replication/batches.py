# Define the fixed cases, systems, and request settings for the five replication batches.

from __future__ import annotations
from copy import deepcopy
from ..demo.bootstrap import bundled_data
from ..pisas import load_scenario
from .inputs import export_pisas, prepare_inputs

OPENROUTER = "https://openrouter.ai/api/v1"
GOOGLE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI = "google/gemini-3.5-flash-lite"
OPUS = "anthropic/claude-opus-5.5"
_CASES = {
    "external-baseline": ("uas_flight_readiness/scenario_02", "uas_flight_readiness/scenario_03",
                          "uas_flight_readiness/scenario_04"),
    "calibration": ("JIRA_allocation/scenario_22", "inpatient_discharge/scenario_07",
                    "manuscript_submission/scenario_13", "meeting_allocation/scenario_07",
                    "outgoing_museum_loan/scenario_11", "severity_classification/scenario_04",
                    "special_event_permit_readiness/scenario_23", "thesis_readiness/scenario_18",
                    "uas_flight_readiness/scenario_13"),
    "wave1": ("JIRA_allocation/scenario_07", "inpatient_discharge/scenario_14",
              "manuscript_submission/scenario_12", "meeting_allocation/scenario_28",
              "outgoing_museum_loan/scenario_25", "severity_classification/scenario_09",
              "special_event_permit_readiness/scenario_07", "thesis_readiness/scenario_22",
              "uas_flight_readiness/scenario_21"),
    "wave2": ("JIRA_allocation/scenario_08", "inpatient_discharge/scenario_02",
              "manuscript_submission/scenario_24", "meeting_allocation/scenario_19",
              "outgoing_museum_loan/scenario_18", "severity_classification/scenario_20",
              "special_event_permit_readiness/scenario_19", "thesis_readiness/scenario_15",
              "uas_flight_readiness/scenario_12"),
    "wave3": ("JIRA_allocation/scenario_11", "inpatient_discharge/scenario_16",
              "manuscript_submission/scenario_02", "meeting_allocation/scenario_25",
              "outgoing_museum_loan/scenario_08", "severity_classification/scenario_23",
              "special_event_permit_readiness/scenario_15", "thesis_readiness/scenario_17",
              "uas_flight_readiness/scenario_20"),
}
_TITLES = {"external-baseline": "External baseline", "calibration": "Calibration",
           "wave1": "Wave 1", "wave2": "Wave 2", "wave3": "Wave 3"}


# Return independent fixed settings so callers cannot change a later batch definition.
def get_batch(name: str) -> dict:
    if name not in _CASES:
        raise ValueError("Unknown batch; choose external-baseline, calibration, wave1, wave2, or wave3")
    late = name in {"wave2", "wave3"}
    external = name == "external-baseline"
    model = OPUS if name == "wave3" else GEMINI
    return {"name": name, "title": _TITLES[name], "cases": _CASES[name],
            "indices": (0, 1) if external else (0,),
            "arms": ("native", "governed-v1", "integrated") if late else ("native", "governed-v1"),
            "actor_model": model, "governed_model": "gemini-3.5-flash-lite" if external else model,
            "governed_base_url": GOOGLE if external else OPENROUTER,
            "governed_key_env": "GOOGLE_API_KEY" if external else "API_KEY",
            "response_contract": late, "top_k": 8,
            "aos_temperature": None, "aos_max_tokens": None, "aos_timeout_seconds": 90,
            "native_temperature": None if name == "wave3" else 0.7,
            "native_max_tokens": 16384, "native_max_rounds": 30, "native_timeout_seconds": 120,
            "judge_profile": "capacity" if late else "legacy",
            "judge_primary_max_tokens": 16384 if late else None,
            "judge_primary_output_policy": "uniform_16384" if late else "upstream_requested",
            "judge_verifier_max_tokens": 1024, "judge_timeout_seconds": 300 if late else 120,
            "judge_maximum_calls": 2048}


# Summarize batch sizes without reading labels or constructing a provider client.
def list_batches() -> list[dict]:
    rows = []
    for name in _CASES:
        batch = get_batch(name)
        rows.append({"name": name, "title": batch["title"], "cases": len(batch["cases"]),
                     "runs": len(batch["cases"]) * len(batch["indices"]) * len(batch["arms"]),
                     "actor_model": batch["actor_model"], "response_contract": batch["response_contract"]})
    return rows


# Reject settings that differ from a named batch's recorded configuration.
def validate_batch(batch: dict) -> dict:
    expected = get_batch(batch.get("name")) if isinstance(batch, dict) else None
    if batch != expected:
        raise ValueError("Batch settings are fixed; use get_batch with a supported name")
    return deepcopy(expected)


# Retain the recorded counterbalancing and the later three-system arm schedule.
def _arm_order(batch: dict, index: int) -> tuple[str, ...]:
    if batch["name"] in {"wave2", "wave3"}:
        pair = ["native", "governed-v1"] if index % 2 == 0 else ["governed-v1", "native"]
        pair.insert(index % 3, "integrated")
        return tuple(pair)
    return ("governed-v1", "native") if index % 2 == 0 else ("native", "governed-v1")


# Enumerate each selected run once with its provider, version, and generation call ceiling.
def planned_runs(batch: dict, inputs: list[dict] | None = None) -> list[dict]:
    batch = validate_batch(batch)
    if inputs is None:
        inputs = [{"case": case, "owner_count": len(load_scenario(
            bundled_data() / "scenarios" / case / "scenario.json")["participants"])}
                  for case in batch["cases"]]
    owners = {row["case"]: row["owner_count"] for row in inputs}
    if (len(inputs) != len(batch["cases"]) or set(owners) != set(batch["cases"])
            or any(type(count) is not int or count < 2 for count in owners.values())):
        raise ValueError("Prepared inputs do not match the selected batch")
    slots = []
    for index, case in enumerate(batch["cases"]):
        for run_index in batch["indices"]:
            for arm in _arm_order(batch, index):
                native = arm == "native"
                model = batch["governed_model"] if arm == "governed-v1" else batch["actor_model"]
                calls = 32 if native else (3 if arm == "integrated" else 2) * owners[case] + 1
                slots.append({"case": case, "arm": arm, "run_index": run_index,
                              "path": f"slots/{case}/{arm}/run{run_index}", "maximum_calls": calls,
                              "model": model,
                              "base_url": batch["governed_base_url"] if arm == "governed-v1" else OPENROUTER,
                              "key_env": batch["governed_key_env"] if arm == "governed-v1" else "API_KEY",
                              "response_contract": batch["response_contract"] and not native,
                              "version": {"native": "native-decentralized", "governed-v1": "governed-v1",
                                          "integrated": "integrated-v3"}[arm]})
    return slots
