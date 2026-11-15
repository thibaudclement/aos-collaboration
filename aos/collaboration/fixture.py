# An explicit offline mechanism fixture (this is never a PiSAs actor model).

from __future__ import annotations
import json
from ..llm import ZERO_USAGE


# Script the authored scheduling fixture to exercise real services without API access.
class FixtureModel:
    # Track fixture calls separately from paid-model usage.
    def __init__(self):
        self.model = "scripted-availability-fixture"
        self.usage = dict(ZERO_USAGE)

    # Propose both an allowed fact and a forbidden fact so the fixture gate must act.
    def complete_json(self, prompt: str, system: str | None = None) -> dict:
        self.usage["calls"] += 1
        data = json.loads(prompt)
        if "PHASE: extract" in (system or ""):
            claims = []
            for unit in data["evidence"]:
                for line in unit["text"].splitlines():
                    if line.strip():
                        claims.append({"text": line, "evidence_ids": [unit["evidence_id"]]})
            return {"claims": claims}
        if "PHASE: review" in (system or ""):
            # Deliberately permissive review tests the independent exact fixture policy.
            return {"decisions": [{"claim_id": claim["claim_id"], "supported": True,
                                   "appropriate": True, "permitted": True}
                                  for claim in data["proposal"]["claims"]]}
        if "PHASE: decide" in (system or ""):
            text = data["gathered_info"]
            if "DEMO_PRIVATE" in text:
                raise AssertionError("Private fixture content reached the decision model")
            enough = all(f"{name} is available from 10:00 to 11:00." in text for name in ("Alex", "Blair"))
            return {"final_response": "Schedule the meeting from 10:00 to 10:30." if enough else "Insufficient evidence.",
                    "used_evidence_refs": data["evidence_refs"] if enough else [],
                    "insufficient_evidence": not enough}
        raise AssertionError("Unknown fixture model phase")
