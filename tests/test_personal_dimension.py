"""M135: a certified dimension Verity marks personal reached mnemiq as an ordinary one.

Verity's D332 gave a dimension a `pii_level` (`none | pii | phi`, this package's own `PII_LEVELS`)
and refuses to slice by a sensitive one in its own compiler. mnemiq's `Dimension` had no such field,
so the classification was dropped on the way in, and `packet.dimensions` -- the CERTIFIED
DIMENSIONS block the generator and the judge both read -- offered a personal dimension like any
other. The field is carried now, and it is applied the way a column's is:

- the packet shows a sensitive dimension only to an identity cleared for its exact level, the
  rule `sql.policy` applies to a column read raw;
- a dimension that is a bare column of its table classifies that column where the column says
  nothing, so the column policy refuses the column itself -- hiding the dimension alone leaves the
  model free to write the column by name.
"""

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import CertifiedRecord, Column, Dimension, Snapshot
from mnemiq.enrichment.certified import apply_certified
from mnemiq.generate.prompts import user_prompt
from mnemiq.semantic.measures import select_dimensions


def _record(dimension_id: str, expr: str, pii_level: str | None) -> CertifiedRecord:
    # The shape Verity's open projection publishes: the dimension's serialised contract, its
    # fields allow-listed by this package's schema.
    payload = {"id": dimension_id, "label": dimension_id.replace("_", " ").title(),
               "source": "customer", "expr": expr}
    if pii_level is not None:
        payload["pii_level"] = pii_level
    return CertifiedRecord.model_validate({
        "envelope": {"object_type": "dimension", "object_id": dimension_id, "version": "v1",
                     "source_system": "verity", "provenance": {"status": "certified",
                                                               "certifier": "reviewer@acme"}},
        "payload": payload,
    })


def _customers(ssn_level: str | None = None) -> Snapshot:
    return Snapshot(version="v", source_id="s", created_at="t", columns=[
        Column(id="customer.ssn", object_id="customer", name="ssn", data_type="text",
               pii_level=ssn_level),
        Column(id="customer.region", object_id="customer", name="region", data_type="text"),
    ])


def _grants(clearance: set[str] = frozenset(), mask: set[str] = frozenset()) -> GrantSet:
    return GrantSet(frozenset({"customer"}), pii_clearance=frozenset(clearance),
                    pii_mask=frozenset(mask))


def test_a_certified_dimension_keeps_its_pii_level():
    out = apply_certified(_customers(), [_record("customer_ssn", "ssn", "pii")])

    assert [(d.id, d.pii_level) for d in out.dimensions] == [("customer_ssn", "pii")]


def test_a_personal_dimension_is_shown_only_to_an_identity_cleared_for_its_level():
    ssn = Dimension(id="customer_ssn", label="SSN", source="customer", expr="ssn", pii_level="pii")
    diagnosis = Dimension(id="diagnosis", label="Diagnosis", source="customer", expr="dx",
                          pii_level="phi")
    region = Dimension(id="region", label="Region", source="customer", expr="region")
    dims = [ssn, diagnosis, region]

    def shown(grants):
        return [d.id for d in select_dimensions(["customer"], dims, grants,
                                                columns=_customers().columns)]

    assert shown(_grants()) == ["region"], "no clearance: the personal ones stay out"
    assert shown(_grants({"pii"})) == ["customer_ssn", "region"], "cleared for pii, not for phi"
    assert shown(_grants({"pii", "phi"})) == ["customer_ssn", "diagnosis", "region"]
    # A masked level is read masked, and a grouping by masked values names no one -- but it is
    # not a raw read, which is what offering the dimension invites.
    assert shown(_grants(mask={"pii"})) == ["region"]


def test_an_unrecognised_dimension_level_reads_as_personal():
    # M41's fail-open, on a dimension: "personal" reads as sensitive to a person and as nothing to
    # a check against the vocabulary. It becomes `pii`, and the run record says which one.
    out = apply_certified(_customers(), [_record("customer_ssn", "lower(ssn)", "personal")])

    assert [d.pii_level for d in out.dimensions] == ["pii"]
    job = next(j for j in out.jobs if j.kind == "certified_pii_level_unrecognised")
    assert job.checkpoints == ["dimension:customer_ssn"]


def _ssn(out) -> Column:
    return next(c for c in out.columns if c.id == "customer.ssn")


def test_the_bare_column_of_a_personal_dimension_requires_its_clearance():
    out = apply_certified(_customers(), [_record("customer_ssn", "SSN", "pii")])

    levels = {c.id: c.pii_levels() for c in out.columns}
    assert levels == {"customer.ssn": {"pii"}, "customer.region": set()}, (
        "unquoted identifiers fold, so `SSN` names the column `ssn`"
    )
    assert _ssn(out).pii_level is None, "the column's own level is its own; the dimension's is beside it"


def test_every_spelling_of_the_column_requires_its_clearance():
    # Each was found leaving `SELECT ssn` open: double quotes and qualifiers by the commit gate,
    # backticks and brackets -- MySQL's, BigQuery's, SQL Server's -- by Codex's review.
    for expr in ('"ssn"', "customer.ssn", 'customer."ssn"', " Customer.SSN ", '"customer"."ssn"',
                 '"customer".ssn', "pg.customer.ssn", "`ssn`", "[ssn]", "customer.`ssn`",
                 "customer.[ssn]", "lower(`ssn`)", "lower([ssn])"):
        out = apply_certified(_customers(), [_record("customer_ssn", expr, "pii")])
        assert _ssn(out).pii_levels() == {"pii"}, expr
    # A column named like a keyword: sqlglot reads `comment` as a command and `case` as nothing.
    for keyword in ("comment", "case", "desc", "customer.comment", '"case"', "customer.select",
                    "customer.not", "customer.true", "customer.false", "pg.customer.select",
                    '"customer".select', "`select`"):
        name = keyword.rpartition(".")[2].strip('"`')
        snap = Snapshot(version="v", source_id="s", created_at="t", columns=[
            Column(id=f"customer.{name}", object_id="customer", name=name)])
        out = apply_certified(snap, [_record("customer_note", keyword, "pii")])
        assert out.columns[0].pii_levels() == {"pii"}, keyword
    # A column of another table names none of this one's.
    out = apply_certified(_customers(), [_record("customer_ssn", "employee.ssn", "pii")])
    assert _ssn(out).pii_levels() == set()


def test_a_quoted_path_names_its_column():
    """Codex's review of option A: BigQuery quotes a whole path, `` `customer.ssn` ``, and sqlite
    reads it as ONE column named `customer.ssn` -- so `ssn` was left open to the uncleared. A name
    read with a dot in it is split as well, and the policy denies the column."""
    from mnemiq.sql.policy import build_access_policy

    for expr in ("`customer.ssn`", "lower(`customer.ssn`)", "`pg.customer.ssn`"):
        out = apply_certified(_customers(), [_record("customer_ssn", expr, "pii")])
        assert _ssn(out).pii_levels() == {"pii"}, expr
        assert ("customer", "ssn") in build_access_policy(out, _grants()).denied, expr
        assert not any(j.kind == "certified_dimension_column_unresolved" for j in out.jobs), (
            f"{expr}: a reference whose split reading is held is resolved")
    # The whole name is kept as a reading too: with a column truly named `customer.ssn` beside
    # `ssn`, the same spelling makes both require the clearance, as both readings could be meant.
    both = Snapshot(version="v", source_id="s", created_at="t", columns=[
        Column(id="customer.ssn", object_id="customer", name="ssn"),
        Column(id="customer.customer.ssn", object_id="customer", name="customer.ssn")])
    out = apply_certified(both, [_record("customer_ssn", "`customer.ssn`", "pii")])
    assert {c.name: c.pii_levels() for c in out.columns} == {
        "ssn": {"pii"}, "customer.ssn": {"pii"}}
    dotted = Snapshot(version="v", source_id="s", created_at="t", columns=[
        Column(id="customer.a.b", object_id="customer", name="a.b")])
    out = apply_certified(dotted, [_record("customer_ab", '"a.b"', "pii")])
    assert out.columns[0].pii_levels() == {"pii"}


def test_every_column_a_personal_expression_reads_requires_its_clearance():
    # A personal value derived from columns makes them personal: `lower(ssn)` left `ssn` open
    # when only an expression that IS a column classified one.
    snap = Snapshot(version="v", source_id="s", created_at="t", columns=[
        Column(id="customer.ssn", object_id="customer", name="ssn"),
        Column(id="customer.first", object_id="customer", name="first"),
        Column(id="customer.last", object_id="customer", name="last"),
        Column(id="customer.region", object_id="customer", name="region")])
    out = apply_certified(snap, [_record("customer_ssn", "lower(ssn)", "pii"),
                                 _record("customer_name", "customer.first || ' ' || last", "pii")])
    assert {c.name for c in out.columns if c.pii_levels()} == {"ssn", "first", "last"}
    # A column the expression only tests takes the level too, by design: refusing more.
    out = apply_certified(snap, [_record("ssn_when", "CASE WHEN region = 'x' THEN ssn END", "pii")])
    assert {c.name for c in out.columns if c.pii_levels()} == {"ssn", "region"}


def test_two_levels_over_one_column_require_both_in_either_order():
    """Codex's review: the first shape kept the first level, so a `pii`-only identity read a
    column a `phi` dimension classified when `pii` arrived first, and was denied when it did not.
    Both clearances are required now, whatever the order."""
    from mnemiq.sql.policy import build_access_policy

    pii, phi = _record("customer_ssn", "ssn", "pii"), _record("customer_dx", "ssn", "phi")
    for records in ([pii, phi], [phi, pii]):
        out = apply_certified(_customers(), records)
        assert _ssn(out).pii_levels() == {"pii", "phi"}

        def disposition(grants):
            policy = build_access_policy(out, grants)
            if ("customer", "ssn") in policy.denied:
                return "denied"
            return "masked" if ("customer", "ssn") in policy.masked else "raw"

        assert disposition(_grants({"pii"})) == "denied"
        assert disposition(_grants({"phi"})) == "denied"
        assert disposition(_grants({"pii", "phi"})) == "raw"
        assert disposition(_grants({"pii"}, mask={"phi"})) == "masked"


def test_a_dimension_adds_to_a_columns_own_level_and_never_lowers_it():
    phi = apply_certified(_customers("phi"), [_record("customer_ssn", "ssn", "pii")])
    assert _ssn(phi).pii_level == "phi"
    assert _ssn(phi).pii_levels() == {"phi", "pii"}

    for record in (_record("customer_ssn", "ssn", None), _record("customer_ssn", "ssn", "none")):
        out = apply_certified(_customers(), [record])
        assert _ssn(out).pii_levels() == set(), record

    # Applied over a column already requiring a dimension's level, the new one is added to it.
    carried = _customers()
    carried.columns[0] = carried.columns[0].model_copy(update={"dimension_pii_levels": ["phi"]})
    out = apply_certified(carried, [_record("customer_ssn", "ssn", "pii")])
    assert _ssn(out).dimension_pii_levels == ["phi", "pii"]


def test_a_personal_dimension_naming_a_column_the_snapshot_lacks_is_recorded():
    # Hidden from the uncleared either way; what the run record says is that some column it reads
    # requires nobody's clearance here: none of its columns held, some of them, or none of its
    # own table's at all.
    out = apply_certified(_customers(), [_record("customer_tin", "tin", "pii"),
                                         _record("customer_ssn", "ssn", "pii"),
                                         _record("customer_dob", "extract(year from dob)", "phi"),
                                         _record("customer_id", "ssn || ' ' || tin", "pii"),
                                         _record("customer_ext", "employee.ssn", "pii")])
    job = next(j for j in out.jobs if j.kind == "certified_dimension_column_unresolved")
    # `customer_id` reads `ssn`, held, and `tin`, not; `customer_ext` reads another table's column.
    assert job.checkpoints == ["customer_dob", "customer_ext", "customer_id", "customer_tin"]
    assert _ssn(out).pii_levels() == {"pii"}
    assert job.status == "refused"


def test_a_column_a_personal_dimension_reads_is_kept_out_of_the_value_index():
    # The value index harvests a bounded string column's distinct values into the model's view;
    # a column personal by its dimension alone was harvested when the gate read the own level.
    from mnemiq.semantic.values import _qualifies

    snap = Snapshot(version="v", source_id="s", created_at="t", columns=[
        Column(id="customer.region", object_id="customer", name="region", data_type="text",
               distinct_count=5)])
    assert _qualifies(snap.columns[0], 200), "setup: an unclassified region would be harvested"
    out = apply_certified(snap, [_record("customer_region", "region", "pii")])
    assert not _qualifies(out.columns[0], 200)


def test_the_card_names_the_levels_a_dimension_gives_a_column():
    from mnemiq.semantic.cards import build_cards

    out = apply_certified(_customers("phi"), [_record("customer_ssn", "ssn", "pii")])
    card = next(c for c in build_cards(out) if c.object_id == "customer").text
    assert "ssn (text, phi, pii)" in card, card


def test_the_column_policy_refuses_the_column_a_personal_dimension_classified():
    """The join to enforcement. Hiding the dimension leaves `SELECT ssn` open to the model; the
    column's classification is what the decider refuses on."""
    from mnemiq.sql.policy import build_access_policy

    out = apply_certified(_customers(), [_record("customer_ssn", "ssn", "pii")])

    uncleared = build_access_policy(out, _grants())
    assert ("customer", "ssn") in uncleared.denied
    assert ("customer", "region") not in uncleared.denied
    assert ("customer", "ssn") not in build_access_policy(out, _grants({"pii"})).denied


def test_the_prompt_offers_a_personal_dimension_only_to_the_cleared(tmp_path):
    """From the record Verity publishes to the prompt the generator reads -- the path the field
    was lost on."""
    from mnemiq.semantic.retrieval import retrieve
    from tests.test_retrieval import FakeEmbedder, _con, _identity

    out = apply_certified(
        Snapshot(version="v", source_id="s", created_at="t"),
        [CertifiedRecord.model_validate({
            "envelope": {"object_type": "dimension", "object_id": "claimant_ssn", "version": "v1",
                         "source_system": "verity", "provenance": {"status": "certified"}},
            "payload": {"id": "claimant_ssn", "label": "Claimant SSN", "source": "claim",
                        "expr": "claimant_ssn", "pii_level": "pii"},
        })],
    )

    class _Authz:
        def __init__(self, grants):
            self._grants = grants

        def grants_for(self, identity):
            return self._grants

    def prompt(grants):
        packet = retrieve(_con(tmp_path), "claim_identifier", _identity(), _Authz(grants),
                          FakeEmbedder(), dimensions=out.dimensions)
        assert [c.object_id for c in packet.cards][:1] == ["claim"], "setup: the claim card"
        return user_prompt(packet)

    assert "Claimant SSN" not in prompt(GrantSet(frozenset({"claim", "policy"})))
    assert "Claimant SSN" in prompt(
        GrantSet(frozenset({"claim", "policy"}), pii_clearance=frozenset({"pii"})))


def test_a_grant_that_clears_everything_clears_a_level_only_a_dimension_carries():
    # `everything` is the widest view, for tests and a single-identity store. Built from the columns'
    # levels alone, it hid a dimension classified in Verity alone -- an expression over a column
    # classifies no column, so the level is on the dimension only.
    from mnemiq.llm.window import everything

    out = apply_certified(_customers(), [_record("customer_ssn", "dob", "phi")])
    assert not any(c.pii_levels() for c in out.columns), "setup: no column carries phi"

    widest = everything(out)
    assert "phi" in widest.pii_clearance
    # A column carrying a dimension's level with the dimension not in this snapshot is cleared too.
    carried = _customers()
    carried.columns[0] = carried.columns[0].model_copy(update={"dimension_pii_levels": ["pii"]})
    assert "pii" in everything(carried).pii_clearance
    assert [d.id for d in select_dimensions(["customer"], out.dimensions, widest,
                                            columns=out.columns)] == ["customer_ssn"]
