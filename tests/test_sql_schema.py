from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.sql.schema import schema_map, visible_schema


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        columns=[
            Column(id="claim.claim_identifier", object_id="claim", name="claim_identifier"),
            Column(id="claim.status", object_id="claim", name="status"),
            Column(id="person.last_name", object_id="person", name="last_name"),
        ],
    )


def test_schema_map_groups_columns_by_table():
    assert schema_map(_snapshot()) == {
        "claim": {"claim_identifier", "status"},
        "person": {"last_name"},
    }


def test_visible_schema_hides_what_is_not_granted():
    visible = visible_schema(_snapshot(), GrantSet(frozenset({"claim"})))
    assert set(visible) == {"claim"}  # person does not exist as far as the decider knows


def test_no_grants_means_no_schema():
    assert visible_schema(_snapshot(), GrantSet(frozenset())) == {}
