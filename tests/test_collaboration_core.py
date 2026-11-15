# Behavioral tests for evidence scope, trusted fixture grants, and canonical release.
# These use synthetic text and scripted review flags. They verify deterministic
# boundaries and deliberately do not claim that a model judges privacy correctly.

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
import pytest
from aos.collaboration.artifacts import ArtifactError, ArtifactIndex, EvidenceBundle
from aos.collaboration.policy import PolicyError, PolicyStore, RequestContext
from aos.collaboration.release import ReleaseError, ReleaseProposal, ReviewRecord, release

RAW = "Morgan is unavailable from 10–11 because of a private medical appointment.\n"
PUBLIC = "Morgan is unavailable from 10–11."
PRIVATE = "Morgan has a private medical appointment."


# Create a real file-backed owner corpus with a policy outside its workspace.
@pytest.fixture
def corpus(tmp_path):
    workspace = tmp_path / "agent-b" / "workspace"
    (workspace / "inputs").mkdir(parents=True)
    source = workspace / "inputs" / "private-calendar.txt"
    source.write_text(RAW, encoding="utf-8")
    index = ArtifactIndex(tmp_path / "private-state" / "evidence.sqlite", workspace, "agent-b")
    units = index.register_file("inputs/private-calendar.txt", "calendar", source="synthetic")
    policy_path = tmp_path / "private-state" / "policy.json"
    policy_path.write_text(json.dumps({"schema_version": 1, "owner_id": "agent-b",
                                      "mode": "inferred", "instructions": "Share availability, never its private reason."}))
    store = PolicyStore(policy_path, workspace=workspace, owner_id="agent-b")
    context = RequestContext("r-1", "agent-a", "agent-b", "schedule", "When is Morgan unavailable?")
    return SimpleNamespace(workspace=workspace, source=source, index=index, units=units,
                           policy_path=policy_path, store=store, context=context)


# Build one proposal using only evidence in the current owner's retrieval bundle.
def _proposal(corpus, texts=(PUBLIC,), *, context=None):
    context = context or corpus.context
    bundle = EvidenceBundle.create(context, corpus.index.search("Morgan"))
    claims = [{"text": text, "evidence_ids": [bundle.units[0].evidence_id]} for text in texts]
    return ReleaseProposal.create(context, bundle, corpus.store.load(), claims)


# Produce a complete scripted review without pretending that it measures a real model.
def _review(proposal, *, approved=True):
    return ReviewRecord.from_dict(proposal, {"decisions": [
        {"claim_id": claim.claim_id, "supported": True, "appropriate": approved,
         "permitted": approved, "reason": "Private review mentioning medical appointment."}
        for claim in proposal.claims]})


# Install a source-version-bound permission for one exact synthetic release view.
def _explicit_policy(corpus):
    grant = {"artifact_id": "calendar", "source_version": corpus.units[0].version,
             "source_text": RAW, "release_text": PUBLIC,
             "recipients": ["agent-a"], "purposes": ["schedule"]}
    corpus.policy_path.write_text(json.dumps({"schema_version": 1, "owner_id": "agent-b",
                                            "mode": "explicit_fixture", "instructions": "Controlled fixture only.",
                                            "grants": [grant]}))


# Preserve all Unicode source characters and exact offsets without silently losing long text.
def test_index_spans_reconstruct_full_source_and_survive_reopen(corpus):
    text = ("A résumé about research.\n" + "Evidence " * 31 + "\n") * 30
    path = corpus.workspace / "inputs" / "long.txt"
    path.write_text(text, encoding="utf-8")
    units = corpus.index.register_file("inputs/long.txt", "long")
    assert len(units) > 1
    assert "".join(unit.text for unit in units) == text
    for unit in units:
        assert text[unit.start:unit.end] == unit.text
        assert unit.version == hashlib.sha256(text.encode("utf-8")).hexdigest()
    reopened = ArtifactIndex(corpus.index.db_path, corpus.workspace, "agent-b")
    assert reopened.register_file("inputs/long.txt", "long") == units


# Keep adjacent evaluator material inaccessible unless an explicit trusted importer registers it.
def test_index_only_reads_explicitly_registered_files(corpus):
    (corpus.workspace / "utility.json").write_text('{"oracle_answer":"GOLD-NEVER-INDEXED"}')
    assert corpus.index.search("GOLD-NEVER-INDEXED") == []
    assert {row["artifact_id"] for row in corpus.index.snapshot()} == {"calendar"}
    assert "GOLD-NEVER-INDEXED" not in json.dumps([unit.as_dict() for unit in corpus.index.search("", limit=None)])


# Find a matching artifact in a collection without manufacturing matches for unrelated questions.
def test_lexical_retrieval_finds_evidence_and_full_context_returns_all(corpus):
    path = corpus.workspace / "inputs" / "distractor.txt"
    path.write_text("The telescope's calibration procedure needs adjustment.")
    corpus.index.register_file("inputs/distractor.txt", "distractor")
    hits = corpus.index.search("Morgan unavailable", limit=1)
    assert [unit.artifact_id for unit in hits] == ["calendar"]
    assert corpus.index.search("zebraquartz") == []
    assert len(corpus.index.search("unmatched", limit=None)) == 2


# Reject both direct path traversal and a symlink that points into another owner's data.
@pytest.mark.parametrize("use_symlink", [False, True], ids=["path-traversal", "symlink-escape"])
def test_registration_cannot_escape_owner_workspace(corpus, tmp_path, use_symlink):
    outside = tmp_path / "other-owner.txt"
    outside.write_text("other owner's secret")
    if use_symlink:
        (corpus.workspace / "inputs" / "escape.txt").symlink_to(outside)
        path = "inputs/escape.txt"
    else:
        path = "../../other-owner.txt"
    with pytest.raises(ArtifactError, match="escape"):
        corpus.index.register_file(path, "escape")


# Prevent an agent-readable index or the reopening of one owner's index under another identity.
def test_index_location_and_owner_are_bound(corpus):
    with pytest.raises(ArtifactError, match="outside"):
        ArtifactIndex(corpus.workspace / "index.sqlite", corpus.workspace, "agent-b")
    with pytest.raises(ArtifactError, match="another owner"):
        ArtifactIndex(corpus.index.db_path, corpus.workspace, "agent-c")
    with pytest.raises(ArtifactError, match="another artifact"):
        corpus.index.register_file("inputs/private-calendar.txt", "alias")


# Detect source edits during discovery, direct evidence access, and final release alike.
@pytest.mark.parametrize("operation", ["search", "read", "release"])
def test_changed_source_invalidates_evidence(corpus, operation):
    proposal = _proposal(corpus)
    review = _review(proposal)
    corpus.source.write_text("Morgan is now available from 10–11.", encoding="utf-8")
    with pytest.raises(ArtifactError, match="changed"):
        if operation == "search":
            corpus.index.search("Morgan")
        elif operation == "read":
            corpus.index.read(corpus.units[0].evidence_id)
        else:
            release(corpus.index, corpus.store, corpus.context, proposal, review)


# Replace obsolete IDs when a source is intentionally reindexed after an edit.
def test_reindexing_cannot_reuse_old_source_evidence(corpus):
    old = corpus.units[0]
    corpus.source.write_text("Morgan is available at noon.")
    updated = corpus.index.register_file("inputs/private-calendar.txt", "calendar")
    assert updated[0].version != old.version
    assert updated[0].evidence_id != old.evidence_id
    with pytest.raises(ArtifactError, match="current index"):
        corpus.index.read(old.evidence_id)


# Honor a narrowed source manifest without deleting the owner's original file.
def test_manifest_removal_prevents_stale_source_retrieval(corpus):
    extra = corpus.workspace / "inputs" / "removed.txt"
    extra.write_text("Restricted astronomical telescope calibration data.")
    units = corpus.index.register_file("inputs/removed.txt", "removed")
    corpus.index.retain_artifacts(["calendar"])
    assert corpus.index.search("astronomical telescope") == []
    with pytest.raises(ArtifactError, match="current index"):
        corpus.index.read(units[0].evidence_id)
    assert extra.is_file()
    reopened = ArtifactIndex(corpus.index.db_path, corpus.workspace, "agent-b")
    assert {row["artifact_id"] for row in reopened.snapshot()} == {"calendar"}


# Support an empty authorized corpus without silently preserving earlier indexed material.
def test_empty_manifest_removes_every_indexed_artifact(corpus):
    corpus.index.retain_artifacts([])
    assert corpus.index.search("", limit=None) == []
    assert corpus.source.is_file()


# Reject a manifest typo atomically instead of pruning valid sources first.
def test_unknown_manifest_entry_does_not_partially_prune(corpus):
    with pytest.raises(ArtifactError, match="unregistered"):
        corpus.index.retain_artifacts(["unknown"])
    assert corpus.index.read(corpus.units[0].evidence_id) == corpus.units[0]


# Reject citations to a valid owner unit that was not retrieved for the present request.
def test_proposal_cannot_cite_outside_request_bundle(corpus):
    bundle = EvidenceBundle.create(corpus.context, [])
    with pytest.raises(ArtifactError, match="outside the request"):
        ReleaseProposal.create(corpus.context, bundle, corpus.store.load(),
                               [{"text": PUBLIC, "evidence_ids": [corpus.units[0].evidence_id]}])


# Reject a different owner's evidence even when the proposed text is superficially harmless.
def test_bundle_cannot_admit_another_owner(corpus):
    foreign = replace(corpus.units[0], owner_id="agent-c")
    with pytest.raises(ArtifactError, match="different owner"):
        EvidenceBundle.create(corpus.context, [foreign])


# Reject unsupported statements, opaque extra output fields, and malformed evidence references.
@pytest.mark.parametrize("claim", [
    {"text": PUBLIC, "evidence_ids": []},
    {"text": PUBLIC, "evidence_ids": "not-a-list"},
    {"text": PUBLIC, "evidence_ids": [], "extra": PRIVATE},
    {"text": "", "evidence_ids": []},
], ids=["no-citation", "wrong-type", "unreviewed-extra-field", "empty-statement"])
def test_proposal_shape_fails_closed(corpus, claim):
    bundle = EvidenceBundle.create(corpus.context, corpus.units)
    with pytest.raises(ReleaseError):
        ReleaseProposal.create(corpus.context, bundle, corpus.store.load(), [claim])


# Bind authorization to every contextual field rather than letting a request ID stand alone.
@pytest.mark.parametrize("changes", [
    {"request_id": "r-2"}, {"requester_id": "agent-c"}, {"owner_id": "agent-c"},
    {"purpose": "publish"}, {"question": "Why is Morgan unavailable?"},
], ids=["request-id", "requester", "owner", "purpose", "question"])
def test_release_cannot_reuse_review_in_another_context(corpus, changes):
    proposal = _proposal(corpus)
    with pytest.raises(ReleaseError, match="different owner or request"):
        release(corpus.index, corpus.store, replace(corpus.context, **changes), proposal, _review(proposal))


# Require typed, complete review decisions instead of treating malformed approval as consent.
@pytest.mark.parametrize("bad", ["true", 1, None], ids=["string", "integer", "null"])
def test_non_boolean_review_approval_is_rejected(corpus, bad):
    proposal = _proposal(corpus)
    value = {"decisions": [{"claim_id": "c1", "supported": True, "appropriate": True, "permitted": bad}]}
    with pytest.raises(ReleaseError, match="JSON booleans"):
        ReviewRecord.from_dict(proposal, value)


# Reject missing, repeated, and fabricated review decisions for a proposed claim.
@pytest.mark.parametrize("ids", [[], ["c1", "c1"], ["c2"]], ids=["missing", "duplicate", "unknown"])
def test_review_must_cover_each_claim_exactly_once(corpus, ids):
    proposal = _proposal(corpus)
    with pytest.raises(ReleaseError, match="exactly once"):
        ReviewRecord.from_dict(proposal, {"decisions": [
            {"claim_id": value, "supported": True, "appropriate": True, "permitted": True}
            for value in ids]})


# Reject an approval if the candidate statement changed after it was reviewed.
def test_review_is_bound_to_exact_proposal_text(corpus):
    original = _proposal(corpus)
    changed = _proposal(corpus, (PRIVATE,))
    with pytest.raises(ReleaseError, match="different proposal"):
        release(corpus.index, corpus.store, corpus.context, changed, _review(original))


# Keep private source content and review explanations out of the deterministic public projection.
def test_explicit_policy_releases_only_approved_sanitized_view(corpus):
    _explicit_policy(corpus)
    proposal = _proposal(corpus, (PUBLIC, PRIVATE))
    review = _review(proposal)
    output = release(corpus.index, corpus.store, corpus.context, proposal, review).as_dict()
    assert output["status"] == "released"
    assert output["answer"] == PUBLIC
    assert output["coverage"] == "not_established"
    assert len(output["claims"]) == 1
    assert set(output["claims"][0]) == {"text", "evidence_refs"}
    assert output["claims"][0]["evidence_refs"][0].startswith("ref-")
    serialized = json.dumps(output)
    for secret in ("medical", "private-calendar", "source_path", "calendar", corpus.units[0].evidence_id,
                   proposal.policy.version, "Private review"):
        assert secret not in serialized
    assert review.as_dict()["proposal_digest"] == proposal.digest()
    assert "medical" in json.dumps(review.as_dict())


# Enforce exact fixture recipients and purposes even when a scripted reviewer approves everything.
@pytest.mark.parametrize("changes", [{"requester_id": "agent-c"}, {"purpose": "publish"}],
                         ids=["unauthorized-recipient", "unauthorized-purpose"])
def test_fixture_permission_is_independent_of_model_approval(corpus, changes):
    _explicit_policy(corpus)
    context = replace(corpus.context, **changes)
    proposal = _proposal(corpus, context=context)
    result = release(corpus.index, corpus.store, context, proposal, _review(proposal)).as_dict()
    assert result["status"] == "unavailable"
    assert result["answer"] == ""
    assert result["claims"] == []


# Keep evidence-reference tokens specific to the request instead of revealing stable private identifiers.
def test_public_evidence_references_are_request_scoped(corpus):
    proposal = _proposal(corpus)
    first = release(corpus.index, corpus.store, corpus.context, proposal, _review(proposal)).as_dict()
    context = replace(corpus.context, request_id="r-2")
    proposal2 = _proposal(corpus, context=context)
    second = release(corpus.index, corpus.store, context, proposal2, _review(proposal2)).as_dict()
    assert first["claims"][0]["evidence_refs"] != second["claims"][0]["evidence_refs"]
    assert first["answer"] == second["answer"]


# Invalidate a prepared approval after a policy revision, even when the source text is unchanged.
def test_policy_change_invalidates_existing_review(corpus):
    proposal = _proposal(corpus)
    data = json.loads(corpus.policy_path.read_text())
    data["instructions"] = "Do not release this owner's information."
    corpus.policy_path.write_text(json.dumps(data))
    with pytest.raises(PolicyError, match="changed"):
        release(corpus.index, corpus.store, corpus.context, proposal, _review(proposal))


# Recheck policy after source reads in case it changes during release validation.
def test_policy_change_during_source_validation_is_caught(corpus, monkeypatch):
    proposal = _proposal(corpus)
    original_read = corpus.index.read

    # Simulate an owner revoking the policy while a source is being validated.
    def read_and_revoke(evidence_id):
        unit = original_read(evidence_id)
        data = json.loads(corpus.policy_path.read_text())
        data["instructions"] = "New owner policy; revoke prior approval."
        corpus.policy_path.write_text(json.dumps(data))
        return unit

    monkeypatch.setattr(corpus.index, "read", read_and_revoke)
    with pytest.raises(PolicyError, match="changed"):
        release(corpus.index, corpus.store, corpus.context, proposal, _review(proposal))


# Prevent agent-editable policy files and reject an inferred policy containing fixture grants.
def test_policy_cannot_be_stored_in_workspace_or_mix_modes(corpus):
    with pytest.raises(PolicyError, match="outside"):
        PolicyStore(corpus.workspace / "policy.json", workspace=corpus.workspace, owner_id="agent-b")
    _explicit_policy(corpus)
    data = json.loads(corpus.policy_path.read_text())
    data["mode"] = "inferred"
    corpus.policy_path.write_text(json.dumps(data))
    with pytest.raises(PolicyError, match="oracle grants"):
        corpus.store.load()


# Recheck the policy's resolved location if a trusted file is later replaced by a workspace symlink.
def test_policy_symlink_cannot_move_trust_into_workspace(corpus):
    inside = corpus.workspace / "policy.json"
    inside.write_text(corpus.policy_path.read_text())
    corpus.policy_path.unlink()
    corpus.policy_path.symlink_to(inside)
    with pytest.raises(PolicyError, match="outside"):
        corpus.store.load()


# Make absence and denial externally indistinguishable so the status does not confirm a private record.
def test_no_evidence_and_withheld_response_share_public_shape(corpus):
    denied = _proposal(corpus)
    denied_output = release(corpus.index, corpus.store, corpus.context, denied, _review(denied, approved=False)).as_dict()
    empty = ReleaseProposal.create(corpus.context, EvidenceBundle.create(corpus.context, []), corpus.store.load(), [])
    empty_output = release(corpus.index, corpus.store, corpus.context, empty, _review(empty)).as_dict()
    assert empty_output == denied_output


# State the inference-mode limitation explicitly: a wrong semantic approval is not detected by structural checks.
def test_inferred_policy_is_not_a_semantic_privacy_oracle(corpus):
    proposal = _proposal(corpus, (PRIVATE,))
    result = release(corpus.index, corpus.store, corpus.context, proposal, _review(proposal)).as_dict()
    assert result["answer"] == PRIVATE
    blocked = release(corpus.index, corpus.store, corpus.context, proposal, _review(proposal, approved=False)).as_dict()
    assert blocked["status"] == "unavailable"


# Refuse a forged evidence snapshot even when it retains a real evidence ID and version.
def test_source_quote_cannot_be_forged_in_private_snapshot(corpus):
    forged = replace(corpus.units[0], text="Fabricated source statement")
    bundle = EvidenceBundle.create(corpus.context, [forged])
    proposal = ReleaseProposal.create(corpus.context, bundle, corpus.store.load(),
                                      [{"text": "Fabricated source statement", "evidence_ids": [forged.evidence_id]}])
    with pytest.raises(ReleaseError, match="changed after"):
        release(corpus.index, corpus.store, corpus.context, proposal, _review(proposal))
