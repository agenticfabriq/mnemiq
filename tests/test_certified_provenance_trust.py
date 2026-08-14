"""M1 -- an unverified trust marker must not open a PII gate.

`apply_certified` used to drop the envelope on its first line and overlay `pii_level` from whatever
arrived. `_qualifies` keeps a column out of the value index when `pii_level` is in `_SENSITIVE_PII`,
so a record setting it to "none" put the column back in scope and `build_value_index` stored every
distinct value. The ordering compounded it: the overlaid column is added to `_protected`, so the LLM
is then forbidden from correcting the level that was just overridden.

The producer half is Verity's D56 -- it hardcoded `"status": "certified"` and left the certifier null
for a whole class of records. Each half is defensible alone; only the seam is a defect.
"""

import logging

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import PII_LEVELS, Column, Snapshot
from mnemiq.contract.records import CertifiedRecord, Provenance, RecordEnvelope
from mnemiq.enrichment.certified import apply_certified
from mnemiq.semantic.values import _qualifies
from mnemiq.sql.policy import build_access_policy


def _snapshot(pii_level: str = "phi") -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-31T00:00:00Z",
        columns=[
            Column(
                id="patient.dx_code",
                object_id="patient",
                name="dx_code",
                data_type="varchar",
                distinct_count=12,
                pii_level=pii_level,
                description="local guess",
            )
        ],
    )


def _record(provenance: Provenance | None, pii_level: str = "none") -> CertifiedRecord:
    return CertifiedRecord.model_validate(
        {
            "envelope": {
                "object_type": "column",
                "object_id": "patient.dx_code",
                "version": "h1",
                "source_system": "verity",
                **({"provenance": provenance.model_dump()} if provenance else {}),
            },
            "payload": {
                "id": "patient.dx_code",
                "object_id": "patient",
                "name": "dx_code",
                "pii_level": pii_level,
                "description": "governed meaning",
                "semantic_type": "code",
            },
        }
    )


_ATTESTED = Provenance(
    status="certified", certifier="reviewer@acme.example", certified_at="2026-07-30T00:00:00Z"
)
# Exactly what Verity published before D56: the status asserted, nobody named.
_UNATTESTED = Provenance(status="certified", certifier=None, certified_at=None)


def test_an_unattested_record_cannot_open_a_pii_gate():
    updated = apply_certified(_snapshot("phi"), [_record(_UNATTESTED, pii_level="none")])
    column = updated.columns[0]

    assert column.pii_level == "phi", (
        "a record asserting certification with nobody attesting it must not weaken a PII level"
    )
    assert not _qualifies(column, max_distinct=200), (
        "and the column must stay out of the value index, which is the actual consequence"
    )


def test_a_record_with_no_provenance_at_all_cannot_open_a_pii_gate():
    updated = apply_certified(_snapshot("pii"), [_record(None, pii_level="none")])

    assert updated.columns[0].pii_level == "pii"


def test_an_unattested_record_still_contributes_meaning():
    # Deliberately narrow: the seam's danger is one field. Refusing whole records would degrade
    # enrichment for a provenance shape Verity emitted for two months.
    updated = apply_certified(_snapshot("phi"), [_record(_UNATTESTED, pii_level="none")])
    column = updated.columns[0]

    assert column.description == "governed meaning"
    assert column.semantic_type == "code"


def test_an_attested_record_may_set_the_level_in_either_direction():
    # Correcting a wrong local guess is exactly what governed meaning is for.
    relaxed = apply_certified(_snapshot("phi"), [_record(_ATTESTED, pii_level="none")])
    assert relaxed.columns[0].pii_level == "none"
    assert _qualifies(relaxed.columns[0], max_distinct=200)

    tightened = apply_certified(_snapshot("none"), [_record(_ATTESTED, pii_level="phi")])
    assert tightened.columns[0].pii_level == "phi"


def test_an_unattested_record_may_still_tighten():
    updated = apply_certified(_snapshot("none"), [_record(_UNATTESTED, pii_level="phi")])

    assert updated.columns[0].pii_level == "phi", "raising sensitivity is never the risk"


def test_a_refused_downgrade_is_recorded_as_a_job_and_logged(caplog):
    # M2's lesson, applied the same day: a silent security downgrade is worse than the downgrade.
    # `snapshot.jobs` is the codebase's own structured run record, so the refusal goes there rather
    # than into a new field invented on the contract.
    with caplog.at_level(logging.WARNING):
        updated = apply_certified(_snapshot("phi"), [_record(_UNATTESTED, pii_level="none")])

    refusals = [job for job in updated.jobs if job.kind == "certified_pii_downgrade_refused"]
    assert len(refusals) == 1, f"no job recorded the refusal: {updated.jobs}"
    assert refusals[0].status == "refused"
    assert "patient.dx_code" in refusals[0].checkpoints

    messages = [record.getMessage() for record in caplog.records]
    assert any("patient.dx_code" in message for message in messages), (
        f"the refusal must name the column: {messages}"
    )


def test_a_clean_overlay_records_no_refusal_job():
    updated = apply_certified(_snapshot("none"), [_record(_ATTESTED, pii_level="none")])

    assert not [job for job in updated.jobs if job.kind == "certified_pii_downgrade_refused"]


# --- M41: the vocabulary is a convention, not an invariant --------------------
# M1 guards the DIRECTION of a level change; nothing guarded the level being a level at all.
# `Column.pii_level` is `str | None`, and this overlay is the one producer that does not clamp
# to PII_LEVELS -- so an off-vocabulary string lands on the column, and `_qualifies` gates on
# `pii_level not in SENSITIVE_PII`, which an unrecognised value passes. It fails OPEN: a level
# that *reads* as sensitive to a human ("personal", "sensitive", a typo'd "pii ") is harvested.


def test_an_off_vocabulary_level_does_not_become_harvestable():
    # The consequence is the assertion: an unrecognised level must never widen the value index.
    updated = apply_certified(_snapshot("pii"), [_record(_ATTESTED, pii_level="personal")])
    column = updated.columns[0]

    assert not _qualifies(column, max_distinct=200), (
        "an unrecognised pii_level must be treated as sensitive, never harvested"
    )


def test_an_off_vocabulary_level_is_not_stored_on_the_column():
    # Storing it would leave every later reader (the clearance check, the card renderer)
    # interpreting a value none of them have a rule for.
    updated = apply_certified(_snapshot("pii"), [_record(_ATTESTED, pii_level="personal")])

    assert updated.columns[0].pii_level in PII_LEVELS


def test_an_off_vocabulary_level_is_recorded_and_logged(caplog):
    # Same rule as the M1 refusal: a silent security-relevant correction is worse than the input.
    with caplog.at_level(logging.WARNING):
        updated = apply_certified(_snapshot("pii"), [_record(_ATTESTED, pii_level="personal")])

    rejects = [job for job in updated.jobs if job.kind == "certified_pii_level_unrecognised"]
    assert len(rejects) == 1, f"no job recorded the unrecognised level: {updated.jobs}"
    assert "patient.dx_code" in rejects[0].checkpoints

    assert any("personal" in record.getMessage() for record in caplog.records), (
        "the log must name the value that was refused"
    )


def test_a_valid_level_records_no_vocabulary_job():
    updated = apply_certified(_snapshot("pii"), [_record(_ATTESTED, pii_level="none")])

    assert not [job for job in updated.jobs if job.kind == "certified_pii_level_unrecognised"]
    assert updated.columns[0].pii_level == "none"  # the M1 path still works


def test_an_unattested_off_vocabulary_level_cannot_move_a_column_off_phi():
    """The clamp must not pre-empt M1's judgement -- found by Codex reviewing the M41 fix.

    Substituting the level BEFORE the downgrade check made the check's own predicate come out
    the other way: `"personal" not in SENSITIVE_PII` is true and refuses, but `"pii"` is in it
    and does not. So an UNATTESTED record could walk a column phi -> pii, which the harvest gate
    does not care about (both are sensitive) and authorization very much does: `pii_clearance`
    grants on the exact level, so a pii-cleared, non-phi-cleared role then reads it raw.
    """
    updated = apply_certified(_snapshot("phi"), [_record(_UNATTESTED, pii_level="personal")])

    assert updated.columns[0].pii_level == "phi", (
        "an unattested record supplying a level we cannot interpret must not change the level"
    )


def test_the_authorization_consequence_of_that_walk_is_closed():
    # The assertion that makes the one above about a boundary rather than a string.
    updated = apply_certified(_snapshot("phi"), [_record(_UNATTESTED, pii_level="personal")])
    policy = build_access_policy(
        updated, GrantSet(frozenset({"patient"}), pii_clearance=frozenset({"pii"}))
    )

    assert ("patient", "dx_code") in policy.denied, (
        "a pii-cleared role must not read a phi column because an unattested record renamed it"
    )


def test_an_attested_off_vocabulary_level_still_clamps():
    # The control for the two above: attestation is what M1 trusts, so the clamp still applies.
    updated = apply_certified(_snapshot("phi"), [_record(_ATTESTED, pii_level="personal")])

    assert updated.columns[0].pii_level == "pii"


def test_an_omitted_level_on_an_attested_record_is_not_clamped():
    # M42's territory and deliberately pinned here: absent is not the same as unrecognised, and
    # the clamp must not start treating silence as an off-vocabulary value.
    updated = apply_certified(_snapshot("none"), [_record(_ATTESTED, pii_level=None)])

    assert updated.columns[0].pii_level is None
    assert not [j for j in updated.jobs if j.kind == "certified_pii_level_unrecognised"]


def test_no_off_vocabulary_value_reaches_a_column_by_any_route():
    """Every combination, asserted rather than eyeballed.

    The two invariants: whatever a record supplies, the level ON the column is always a real
    level, and an off-vocabulary input never leaves the column harvestable. Written as a sweep
    because the interesting cases are combinations -- the branch taken depends on the local
    level, the supplied value AND attestation together.

    Honest about its limits: this sweep would NOT have caught the regression that made it worth
    writing. That one walked `phi` to `pii`, which is in-vocabulary and not harvestable, and only
    shows up as a difference in what `pii_clearance` grants. The named tests above cover it. A
    property test pins the class; it does not replace knowing which case matters.
    """
    off_vocabulary = ["personal", "sensitive", "pii ", "PII", "high", "", "confidential"]

    for start in (None, "none", "pii", "phi"):
        for supplied in off_vocabulary:
            for provenance in (_ATTESTED, _UNATTESTED):
                updated = apply_certified(
                    _snapshot(start), [_record(provenance, pii_level=supplied)]
                )
                column = updated.columns[0]

                assert column.pii_level in PII_LEVELS, (
                    f"{supplied!r} reached the column (start={start!r}, "
                    f"attested={provenance is _ATTESTED})"
                )
                assert not _qualifies(column, max_distinct=200), (
                    f"{supplied!r} left the column harvestable (start={start!r})"
                )


def test_both_kinds_of_refusal_survive_the_same_run():
    """Two columns, two different refusals, one call -- both must be recorded.

    Each writes `updates["jobs"]` from `snapshot.jobs`, so the second assignment can rebuild the
    list without the first and drop it silently. The logs would still show both, which is exactly
    the failure that looks fine while the structured record -- the durable half (M2) -- is gone.
    """
    snapshot = Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-31T00:00:00Z",
        columns=[
            Column(id="t.a", object_id="t", name="a", data_type="varchar",
                   distinct_count=5, pii_level="phi"),
            Column(id="t.b", object_id="t", name="b", data_type="varchar",
                   distinct_count=5, pii_level="phi"),
        ],
    )

    def record(object_id: str, pii_level: str, provenance: Provenance) -> CertifiedRecord:
        return CertifiedRecord.model_validate({
            "envelope": {"object_type": "column", "object_id": object_id, "version": "h1",
                         "source_system": "verity", "provenance": provenance.model_dump()},
            "payload": {"id": object_id, "object_id": "t", "name": object_id.split(".")[1],
                        "pii_level": pii_level},
        })

    updated = apply_certified(snapshot, [
        record("t.a", "none", _UNATTESTED),        # refused downgrade
        record("t.b", "personal", _ATTESTED),      # unrecognised level
    ])

    kinds = {job.kind for job in updated.jobs}
    assert "certified_pii_downgrade_refused" in kinds, kinds
    assert "certified_pii_level_unrecognised" in kinds, kinds
