"""M4 + M5 -- the channels that disclose must consult the column policy.

`build_access_policy` resolves `denied`/`masked`/`row_filters` per column and is correct. Its two
production callers were `plan_query` and `Runtime.write` -- neither a channel that returns data to a
user. So:

* the schema card, served by `retrieve` and `Runtime.schema`, was rendered once at enrich time from
  the whole snapshot and listed every column of a granted table with its pii_level and its harvested
  coded values (M4);
* the value-grounding refusal quoted `"the real values include: ..."` from an index built pre-RLS,
  before the row-filter rewrite ran (M5).

Both are one shape: the policy exists, and the disclosing channel does not ask it.
"""

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.contract.values import CodedValue
from mnemiq.semantic.cards import build_cards

_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-31T00:00:00Z",
        columns=[
            Column(
                id="patient.mrn",
                object_id="patient",
                name="mrn",
                data_type="varchar",
                pii_level="phi",
                description="medical record number",
                coded_values=[CodedValue(code="A1", meaning="cohort A")],
            ),
            Column(
                id="patient.dx_code",
                object_id="patient",
                name="dx_code",
                data_type="varchar",
                pii_level="pii",
                description="diagnosis code",
                coded_values=[CodedValue(code="E11", meaning="type 2 diabetes")],
            ),
            Column(
                id="patient.visit_count",
                object_id="patient",
                name="visit_count",
                data_type="integer",
                pii_level="none",
                description="visits in period",
            ),
        ],
    )


def _card_text(snapshot: Snapshot, grants: GrantSet) -> str:
    from mnemiq.sql.policy import build_access_policy

    policy = build_access_policy(snapshot, grants)
    cards = build_cards(snapshot, policy=policy)
    return next(card.text for card in cards if card.object_id == "patient")


# grants `patient`, no PII clearance at all -> both sensitive columns are denied
_NO_CLEARANCE = GrantSet(frozenset({"patient"}))
# same, but phi is masked rather than denied
_PHI_MASKED = GrantSet(frozenset({"patient"}), pii_mask=frozenset({"phi"}))
_FULL = GrantSet(
    frozenset({"patient"}), pii_clearance=frozenset({"pii", "phi"})
)


def test_a_denied_column_does_not_appear_on_the_card_at_all():
    text = _card_text(_snapshot(), _NO_CLEARANCE)

    assert "visit_count" in text, "a column the identity may read stays"
    assert "mrn" not in text, "a denied column must not be named"
    assert "dx_code" not in text
    assert "diagnosis code" not in text, "nor its description"
    assert "type 2 diabetes" not in text, "nor its harvested values -- that is the data itself"


def test_a_masked_column_keeps_its_name_and_type_and_loses_the_data():
    text = _card_text(_snapshot(), _PHI_MASKED)

    assert "mrn" in text, "a masked column can be named -- that is what masking means"
    assert "varchar" in text
    assert "medical record number" not in text, "but nothing derived from the data survives"
    assert "cohort A" not in text
    # dx_code is `pii`, which is neither cleared nor masked here, so it stays denied.
    assert "dx_code" not in text


def test_full_clearance_is_unchanged():
    text = _card_text(_snapshot(), _FULL)

    for expected in ("mrn", "dx_code", "visit_count", "diagnosis code", "type 2 diabetes"):
        assert expected in text, f"{expected} missing for an identity cleared to see everything"


def test_no_policy_renders_exactly_what_it_always_did():
    # build_cards is called at enrich time with no identity in play; that path must not change.
    snapshot = _snapshot()
    assert build_cards(snapshot) == build_cards(snapshot, policy=None)


# --------------------------------------------------------------------------------------------
# The serving channels. A correct renderer that nothing calls is the shape this finding IS.
# --------------------------------------------------------------------------------------------


class _FixedAuthz:
    def __init__(self, grants: GrantSet):
        self._grants = grants

    def grants_for(self, identity):
        return self._grants


def _runtime_with(grants: GrantSet):
    from mnemiq.runtime import Runtime

    runtime = Runtime.__new__(Runtime)  # no adapter, no LLM: only schema() is under test
    runtime.snapshot = _snapshot()
    runtime.authz = _FixedAuthz(grants)
    import duckdb

    runtime.con = duckdb.connect()
    runtime.con.execute("CREATE TABLE semantic_object (object_id VARCHAR, card VARCHAR)")
    for card in build_cards(runtime.snapshot):  # stored unscoped, as at enrich time
        runtime.con.execute(
            "INSERT INTO semantic_object VALUES (?, ?)", [card.object_id, card.text]
        )
    return runtime


def test_runtime_schema_scopes_the_card_it_serves():
    # Its docstring promised "access-scoped, so metadata never leaks" while filtering by table only.
    rows = _runtime_with(_NO_CLEARANCE).schema(_IDENTITY)

    card = next(row["card"] for row in rows if row["object_id"] == "patient")
    assert "visit_count" in card
    assert "mrn" not in card, "the stored card lists it; the served one must not"
    assert "type 2 diabetes" not in card


def test_runtime_schema_is_unchanged_for_a_cleared_identity():
    rows = _runtime_with(_FULL).schema(_IDENTITY)

    card = next(row["card"] for row in rows if row["object_id"] == "patient")
    assert "mrn" in card and "type 2 diabetes" in card


def test_retrieve_scopes_the_cards_it_puts_in_the_packet(tmp_path):
    # The channel that matters most: these cards are what the model reads to write SQL.
    from mnemiq.llm.embeddings import FakeEmbedder
    from mnemiq.semantic.retrieval import retrieve
    from mnemiq.semantic.store import build_index
    from mnemiq.store.bootstrap import init_store

    snapshot = _snapshot()
    con = init_store(str(tmp_path / "s.duckdb"))
    build_index(con, snapshot, FakeEmbedder())  # stored unscoped, as at enrich time

    packet = retrieve(
        con,
        "how many visits",
        _IDENTITY,
        _FixedAuthz(_NO_CLEARANCE),
        FakeEmbedder(),
        columns=snapshot.columns,
        snapshot=snapshot,
    )

    assert packet.cards, "the granted table is still retrieved"
    text = "\n".join(card.card for card in packet.cards)
    assert "visit_count" in text
    assert "mrn" not in text, "the model must not be shown a column the identity cannot read"
    assert "type 2 diabetes" not in text


# --------------------------------------------------------------------------------------------
# M5 -- the value-grounding refusal ran BEFORE the RLS rewrite, against an index built with no
# predicate, and quoted what it found. A row-filtered identity could enumerate a bounded column
# without executing anything.
# --------------------------------------------------------------------------------------------


class _Values:
    """The pre-RLS index: it holds every distinct value in the source, unfiltered."""

    def has(self, table, column):
        return (table, column) == ("patient", "region")

    def contains(self, table, column, literal):
        return literal in {"west", "east"}

    def nearest(self, table, column, literal):
        return ["west", "east"]


def _refusal_for(row_filters):
    import sqlglot

    from mnemiq.sql.values_check import check_values

    ast = sqlglot.parse_one("SELECT count(*) FROM patient WHERE region = 'nrth'")
    return check_values(
        ast,
        {"patient": {"region"}},
        _Values(),
        row_filtered={t for t in row_filters},
    )


def test_a_row_filtered_identity_is_refused_without_being_shown_the_values():
    refusal = _refusal_for({"patient"})

    assert refusal is not None, "the accuracy check still fires -- that is not what changed"
    assert "nrth" in refusal.message, "and still names the literal that is wrong"
    assert "west" not in refusal.message, (
        "but must not quote values drawn from rows this identity cannot see"
    )
    assert "east" not in refusal.message


def test_an_unfiltered_identity_still_gets_the_values():
    refusal = _refusal_for(set())

    assert refusal is not None
    assert "west" in refusal.message, "the repair hint is the whole point when nothing is hidden"
