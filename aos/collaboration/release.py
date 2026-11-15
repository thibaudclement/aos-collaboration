# Bind private evidence and review decisions to a canonical outward response.
# Structural checks are deterministic. In inferred mode, supported/appropriate/
# permitted judgments come from a model and can be wrong. This module does not turn
# those judgments into a proof of semantic privacy or retrieval completeness.

from __future__ import annotations
from dataclasses import asdict, dataclass
from .artifacts import ArtifactIndex, EvidenceBundle, canonical_hash
from .policy import PolicySnapshot, PolicyStore, RequestContext

MAX_CLAIMS = 32
MAX_CLAIM_CHARS = 2_000
MAX_RESPONSE_CHARS = 16_000


# Signal a malformed proposal, mismatched review, or invalid release attempt.
class ReleaseError(ValueError):
    pass


# Preserve one proposed statement and the request-local evidence offered for it.
@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]

    # Export the private claim representation used by extraction and review.
    def as_dict(self) -> dict:
        return {"claim_id": self.claim_id, "text": self.text,
                "evidence_ids": list(self.evidence_ids)}


# Freeze a complete proposal before the separate semantic review is requested.
@dataclass(frozen=True)
class ReleaseProposal:
    context_hash: str
    policy: PolicySnapshot
    bundle: EvidenceBundle
    claims: tuple[Claim, ...]

    # Validate claim shape and citations without inventing support for their meaning.
    @classmethod
    def create(cls, context: RequestContext, bundle: EvidenceBundle, policy: PolicySnapshot,
               claims: object) -> ReleaseProposal:
        if bundle.context_hash != context.digest() or bundle.owner_id != context.owner_id:
            raise ReleaseError("The evidence bundle belongs to a different request context.")
        if policy.owner_id != context.owner_id:
            raise ReleaseError("The policy belongs to a different owner.")
        if not isinstance(claims, list) or len(claims) > MAX_CLAIMS:
            raise ReleaseError(f"claims must be a list of at most {MAX_CLAIMS} statements.")
        parsed = []
        for number, value in enumerate(claims, 1):
            if not isinstance(value, dict) or set(value) != {"text", "evidence_ids"}:
                raise ReleaseError("A proposed claim requires text and evidence_ids only.")
            text = value["text"]
            if (not isinstance(text, str) or not text.strip() or len(text) > MAX_CLAIM_CHARS
                    or "\0" in text):
                raise ReleaseError(f"Claim text must contain 1 to {MAX_CLAIM_CHARS} characters.")
            ids = value["evidence_ids"]
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 12
                    or any(not isinstance(item, str) for item in ids)
                    or len(ids) != len(set(ids))):
                raise ReleaseError("Each claim requires 1 to 12 unique evidence IDs.")
            for evidence_id in ids:
                bundle.get(evidence_id)
            parsed.append(Claim(f"c{number}", text, tuple(ids)))
        if sum(len(claim.text) for claim in parsed) > MAX_RESPONSE_CHARS:
            raise ReleaseError("The proposed response exceeds the 16000-character limit.")
        return cls(context.digest(), policy, bundle, tuple(parsed))

    # Hash the exact content, source snapshots, policy revision, and request context.
    def digest(self) -> str:
        return canonical_hash({"context_hash": self.context_hash, "policy": self.policy.as_dict(),
                               "bundle": self.bundle.as_dict(),
                               "claims": [claim.as_dict() for claim in self.claims]})

    # Export the private proposal so the reviewer sees exactly what it is judging.
    def as_dict(self) -> dict:
        return {"digest": self.digest(), "context_hash": self.context_hash,
                "policy_version": self.policy.version,
                "claims": [claim.as_dict() for claim in self.claims],
                "evidence": self.bundle.as_dict()}


# Keep every semantic review flag separate from the resulting public message.
@dataclass(frozen=True)
class ClaimReview:
    claim_id: str
    supported: bool
    appropriate: bool
    permitted: bool
    reason: str = ""

    # Require all three semantic judgments before a statement becomes a candidate for release.
    def approved(self) -> bool:
        return self.supported and self.appropriate and self.permitted


# Bind parsed model judgments to the exact proposal that was reviewed.
@dataclass(frozen=True)
class ReviewRecord:
    proposal_digest: str
    decisions: tuple[ClaimReview, ...]

    # Reject incomplete reviews, unknown claims, duplicates, and truthy non-boolean flags.
    @classmethod
    def from_dict(cls, proposal: ReleaseProposal, value: object) -> ReviewRecord:
        if not isinstance(value, dict) or set(value) != {"decisions"} or not isinstance(value["decisions"], list):
            raise ReleaseError("A review requires a decisions list only.")
        expected = {claim.claim_id for claim in proposal.claims}
        seen = set()
        decisions = []
        for entry in value["decisions"]:
            required = {"claim_id", "supported", "appropriate", "permitted"}
            if not isinstance(entry, dict) or not required <= set(entry) or set(entry) - required - {"reason"}:
                raise ReleaseError("A review decision has missing or unexpected fields.")
            claim_id = entry["claim_id"]
            if not isinstance(claim_id, str) or claim_id not in expected or claim_id in seen:
                raise ReleaseError("Every proposed claim must be reviewed exactly once.")
            if any(type(entry[key]) is not bool for key in ("supported", "appropriate", "permitted")):
                raise ReleaseError("Review flags must be JSON booleans, not strings or integers.")
            reason = entry.get("reason", "")
            if not isinstance(reason, str) or len(reason) > 4_000:
                raise ReleaseError("Private review reasons must be strings of at most 4000 characters.")
            decisions.append(ClaimReview(claim_id, entry["supported"], entry["appropriate"],
                                         entry["permitted"], reason))
            seen.add(claim_id)
        if seen != expected:
            raise ReleaseError("Every proposed claim must be reviewed exactly once.")
        return cls(proposal.digest(), tuple(decisions))

    # Export private review evidence without mixing it into the public answer.
    def as_dict(self) -> dict:
        return {"proposal_digest": self.proposal_digest,
                "decisions": [asdict(decision) for decision in self.decisions]}


# Expose only approved text and request-scoped opaque evidence references.
@dataclass(frozen=True)
class ReleasedClaim:
    text: str
    evidence_refs: tuple[str, ...]

    # Serialize a citation without a source path, original quote, or private artifact ID.
    def as_dict(self) -> dict:
        return {"text": self.text, "evidence_refs": list(self.evidence_refs)}


# Keep public release results independent of private denial reasons and proposal counts.
@dataclass(frozen=True)
class ReleasedResponse:
    request_id: str
    claims: tuple[ReleasedClaim, ...]

    # Build the response entirely from released claims rather than unreviewed model prose.
    def as_dict(self) -> dict:
        return {"request_id": self.request_id,
                "status": "released" if self.claims else "unavailable",
                "answer": "\n".join(claim.text for claim in self.claims),
                "claims": [claim.as_dict() for claim in self.claims],
                "coverage": "not_established"}


# Revalidate all bindings immediately before constructing the only permitted outward payload.
def release(index: ArtifactIndex, policy_store: PolicyStore, context: RequestContext,
            proposal: ReleaseProposal, review: ReviewRecord) -> ReleasedResponse:
    if index.owner_id != context.owner_id or proposal.context_hash != context.digest():
        raise ReleaseError("A release cannot be reused for a different owner or request.")
    if proposal.bundle.context_hash != context.digest() or proposal.bundle.owner_id != context.owner_id:
        raise ReleaseError("The retrieved evidence is bound to a different request.")
    if review.proposal_digest != proposal.digest():
        raise ReleaseError("The review is bound to a different proposal.")
    expected = {claim.claim_id for claim in proposal.claims}
    if len(review.decisions) != len(expected) or {item.claim_id for item in review.decisions} != expected:
        raise ReleaseError("The review does not cover the exact proposed claims.")
    policy_store.assert_current(proposal.policy)
    decisions = {item.claim_id: item for item in review.decisions}
    checked = {}
    for claim in proposal.claims:
        for evidence_id in claim.evidence_ids:
            if evidence_id not in checked:
                current = index.read(evidence_id)
                if current != proposal.bundle.get(evidence_id):
                    raise ReleaseError("Evidence changed after the proposal was prepared.")
                checked[evidence_id] = current
    policy_store.assert_current(proposal.policy)
    approved = []
    for claim in proposal.claims:
        units = [checked[evidence_id] for evidence_id in claim.evidence_ids]
        if decisions[claim.claim_id].approved() and proposal.policy.allows(context, claim.text, units):
            refs = tuple("ref-" + canonical_hash([context.digest(), evidence_id])[:24]
                         for evidence_id in claim.evidence_ids)
            approved.append(ReleasedClaim(claim.text, refs))
    return ReleasedResponse(context.request_id, tuple(approved))
