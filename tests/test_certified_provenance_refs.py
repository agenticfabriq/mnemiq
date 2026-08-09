"""M24 — the snapshot remembers WHICH certified version it applied.

`apply_certified` appended `rec.payload` and dropped `rec.envelope`, where the certified version
hash lives. So mnemiq carried the CONTENT of certified records and could not say which VERSION
produced it.

That is Verity D131's mirror image. D131 was a foreign key with no writer and no reader, recording
which certified records an answer relied on; mnemiq never kept the reference at all. Between them
the chain the product's claim rests on -- *which certified meaning did this answer use* -- had
neither end, and neither end failed loudly.

The refs are provenance, not content: they are deliberately NOT folded into `content_version`, so
two snapshots asserting the same meaning from different record versions still share a cache entry.
"""

from mnemiq.contract.records import CertifiedRecord, RecordEnvelope
from mnemiq.contract.semantic import Definition, MeasureExpr, Metric, Snapshot
from mnemiq.enrichment.certified import apply_certified
from mnemiq.enrichment.pipeline import content_version


def _snapshot(**kw) -> Snapshot:
    return Snapshot(version="v1", source_id="src", created_at="2026-08-09T00:00:00Z", **kw)


def _definition(object_id: str, version: str) -> CertifiedRecord:
    return CertifiedRecord(
        envelope=RecordEnvelope(object_type="definition", object_id=object_id,
                                version=version, source_system="verity"),
        payload=Definition(id=object_id, term=object_id, domain="ops", definition="d"),
    )


def _metric(object_id: str, version: str) -> CertifiedRecord:
    return CertifiedRecord(
        envelope=RecordEnvelope(object_type="metric", object_id=object_id,
                                version=version, source_system="verity"),
        payload=Metric(id=object_id, label=object_id, status="certified", owner="finance",
                       grain="day", time_dimension="d.day",
                       measure=MeasureExpr(expr="count(*)", source="fact")),
    )


def test_an_applied_record_leaves_its_version_behind():
    snap = apply_certified(_snapshot(), [_definition("loss_ratio", "sha256:aaa")])

    assert len(snap.certified_refs) == 1
    ref = snap.certified_refs[0]
    assert (ref.object_type, ref.object_id, ref.version_hash) == (
        "definition", "loss_ratio", "sha256:aaa",
    )


def test_a_namesake_across_types_is_two_refs_not_one():
    """D124's lesson, one repo over: `object_id` does not identify a record.

    `loss_ratio` is legitimately both the metric (the computation) and the definition (the glossary
    entry). Keying provenance on the id alone would silently collapse them and attribute an answer
    to whichever was applied last."""
    snap = apply_certified(
        _snapshot(),
        [_definition("loss_ratio", "sha256:def"), _metric("loss_ratio", "sha256:met")],
    )

    by_type = {r.object_type: r.version_hash for r in snap.certified_refs}
    assert by_type == {"definition": "sha256:def", "metric": "sha256:met"}


def test_no_records_means_no_refs():
    """The bare arm of the ablation must be able to say it relied on nothing, and say it as an
    empty list rather than as an absence."""
    snap = apply_certified(_snapshot(), [])

    assert snap.certified_refs == []


def test_refs_do_not_change_the_content_version():
    """Provenance is not content. Two snapshots asserting the same meaning from different record
    versions must still share a cache entry, or every re-certification invalidates every
    downstream enrichment for no semantic reason."""
    bare = _snapshot()
    grounded = apply_certified(_snapshot(), [_definition("loss_ratio", "sha256:aaa")])
    other_version = apply_certified(_snapshot(), [_definition("loss_ratio", "sha256:bbb")])

    assert content_version(grounded) == content_version(other_version)
    assert content_version(grounded) == content_version(bare), (
        "a standalone definition is not part of content_version's body today; if that changes, "
        "change it deliberately rather than as a side effect of adding provenance"
    )


def test_the_refs_survive_a_round_trip():
    """The snapshot is persisted and reloaded (`save_snapshot`/`load_snapshot`), so provenance that
    does not serialize is provenance that disappears on the next `ask`."""
    snap = apply_certified(_snapshot(), [_definition("loss_ratio", "sha256:aaa")])

    restored = Snapshot.model_validate_json(snap.model_dump_json(by_alias=True))

    assert [r.version_hash for r in restored.certified_refs] == ["sha256:aaa"]
