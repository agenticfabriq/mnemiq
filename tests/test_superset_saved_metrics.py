"""The saved-metric catalogue: read as the builder over Superset's MCP service, with the extension's certification
beside it, reused when a planned measure says what a saved metric already says, and carried onto a join.

Runs against `FakeSuperset`, a real MCP server over streamable HTTP, through the real client.
"""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetResponseTooLarge
from mnemiq.superset.extension import TOOL_PREFIX, ExtensionMissing, call_tool
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.saved_metrics import (
    UNKNOWN,
    Carried,
    SavedMetric,
    carry,
    normalized,
    read_saved_metrics,
    reusable,
)
from tests.superset_fake import FakeSuperset

DATASET_METRICS = TOOL_PREFIX + "dataset_metrics"
USD = {"symbol": "USD", "symbolPosition": "prefix"}
TOO_LARGE = ("Response too large: 61,234 bytes (limit: 50,000)\n\nThis response would overwhelm the LLM context window.\n"
             "Tool: list_metrics")


@pytest.fixture
def fake():
    with FakeSuperset() as served:
        yield served


def _client(fake) -> SupersetMcp:
    return SupersetMcp(fake.url, DelegatedToken("tok"))


def _metric(dataset_id: int, name: str, expression: str | None = None, **more) -> dict:
    return {"name": name, "verbose_name": more.get("verbose_name"), "description": None,
            "expression": expression or f"SUM({name})", "d3format": more.get("d3format"),
            "warning_text": more.get("warning_text"), "source": "builtin", "dataset_id": dataset_id,
            "dataset_name": f"t{dataset_id}", "view_id": None, "view_name": None, "compatible_dimensions": []}


def _serve_metrics(fake, by_dataset: dict[int, list[dict]]) -> None:
    """Answer `list_metrics` as rc2 does: sorted, paged, with the totals."""
    def answer(request, authorization):
        rows = by_dataset.get(request["dataset_id"], [])
        size, page = request["page_size"], request["page"]
        return {"metrics": rows[(page - 1) * size: page * size], "total_count": len(rows), "page": page,
                "page_size": size, "total_pages": max(1, -(-len(rows) // size)), "success": True}

    fake.reply("list_metrics", answer)


def _serve_extension(fake, by_dataset: dict[int, list[dict]]) -> None:
    def answer(request, authorization):
        return {"datasets": [{"dataset_id": i, "metrics": by_dataset[i]} for i in request["dataset_ids"]
                             if i in by_dataset]}

    fake.reply(DATASET_METRICS, answer)


def _extension_row(name: str, **more) -> dict:
    return {"metric_name": name, "expression": more.get("expression", f"SUM({name})"), "d3format": None,
            "currency": more.get("currency"), "certified_by": more.get("certified_by"),
            "certification_details": None, "warning_text": None}


def _measure(expression: str, dataset_id: int = 1, name: str = "revenue") -> Measure:
    return Measure(name=name, label="Revenue", expression=expression, kind=MeasureKind.NUMBER,
                   dataset_id=dataset_id, description="", asked_by="west")


def _saved(name: str, expression: str, *, dataset_id: int = 1, certified_by=None) -> SavedMetric:
    return SavedMetric(dataset_id=dataset_id, name=name, expression=expression, verbose_name=None, d3format=None,
                       currency=None, certified_by=certified_by, warning_text=None)


def test_list_metrics_paged_by_dataset(fake):
    many = [_metric(1, f"m{i:02d}") for i in range(30)]
    _serve_metrics(fake, {1: many, 2: [_metric(2, "total", "SUM(x)", d3format=",d", verbose_name="Total")]})
    _serve_extension(fake, {1: [_extension_row("m00", currency=USD, certified_by="finance")]
                            + [_extension_row(m["name"]) for m in many[1:]],
                            2: [_extension_row("total")]})
    saved = read_saved_metrics(_client(fake), dataset_ids=(1, 2))
    listed = [c.request for c in fake.calls if c.tool == "list_metrics"]
    assert listed == [{"dataset_id": 1, "page_size": 25, "page": 1}, {"dataset_id": 1, "page_size": 25, "page": 2},
                      {"dataset_id": 2, "page_size": 25, "page": 1}]
    assert len(saved) == 31
    first = next(s for s in saved if (s.dataset_id, s.name) == (1, "m00"))
    assert first.currency == USD and first.certified_by == "finance"
    plain = next(s for s in saved if (s.dataset_id, s.name) == (1, "m01"))
    assert plain.currency is None and plain.certified_by is None
    total = next(s for s in saved if s.dataset_id == 2)
    assert (total.verbose_name, total.d3format, total.expression) == ("Total", ",d", "SUM(x)")
    assert [c.request for c in fake.calls if c.tool == DATASET_METRICS] == [{"dataset_ids": [1, 2]}]


def test_the_extension_is_asked_twenty_datasets_at_a_time():
    """The extension's `dataset_metrics` takes at most 20 dataset ids a call and refuses more
    (`reads.MAX_DATASETS`); 45 datasets are asked in calls of 20, 20 and 5, in order, and every metric's
    certification is known."""
    from tests.superset_extension_path import use_extension_source

    use_extension_source()
    from agenticfabriq.mnemiq import reads

    from mnemiq.superset import extension

    ids = tuple(range(101, 146))
    with FakeSuperset() as fake:
        _serve_metrics(fake, {i: [_metric(i, f"m{i}")] for i in ids})

        def answer(request, authorization):
            if len(request["dataset_ids"]) > reads.MAX_DATASETS:
                return {"refused": "request", "field": "dataset_ids"}
            return {"datasets": [{"dataset_id": i, "metrics": [_extension_row(f"m{i}", certified_by="finance")]}
                                 for i in request["dataset_ids"]]}

        fake.reply(DATASET_METRICS, answer)
        saved = read_saved_metrics(_client(fake), dataset_ids=ids)
        asked = [c.request["dataset_ids"] for c in fake.calls if c.tool == DATASET_METRICS]
    assert asked == [list(ids[:20]), list(ids[20:40]), list(ids[40:])]
    assert len(saved) == 45 and all(s.certified_by == "finance" for s in saved)
    # mnemiq's batch is the extension's own limit.
    assert extension.DATASET_METRICS_BATCH == reads.MAX_DATASETS


def test_without_the_extension_certification_is_unknown():
    with FakeSuperset(extension=False) as fake:
        _serve_metrics(fake, {1: [_metric(1, "a"), _metric(1, "b")]})
        saved = read_saved_metrics(_client(fake), dataset_ids=(1,))
        extension_calls = [c.tool for c in fake.calls if c.tool.startswith("extensions.")]
    assert [s.name for s in saved] == ["a", "b"]
    assert all(s.certified_by is UNKNOWN and s.currency is UNKNOWN for s in saved)
    # The call and its one probe, and nothing retried after.
    assert extension_calls == [DATASET_METRICS, DATASET_METRICS]
    # With nothing known about certification, none is preferred: the order is by name.
    plain = [_saved("b", "SUM(amount)", certified_by=UNKNOWN), _saved("a", "SUM(amount)", certified_by=UNKNOWN)]
    assert reusable(_measure("sum(amount)"), plain).name == "a"


def test_a_missing_extension_is_told_from_a_failing_one():
    with FakeSuperset() as fake:
        # The tool exists and fails, but the probe with no datasets answers: the failure is the tool's own.
        def answer(request, authorization):
            if request["dataset_ids"]:
                raise RuntimeError(f"Internal error in {DATASET_METRICS}: An unexpected error occurred. Error ID: 1.")
            return {"datasets": []}

        fake.reply(DATASET_METRICS, answer)
        with pytest.raises(Exception) as caught:
            call_tool(_client(fake), "dataset_metrics", {"dataset_ids": [1]})
        assert not isinstance(caught.value, ExtensionMissing)
    with FakeSuperset(extension=False) as fake:
        with pytest.raises(ExtensionMissing) as caught:
            call_tool(_client(fake), "dataset_metrics", {"dataset_ids": [1]})
        assert "Error ID" not in str(caught.value)
        assert "Error ID" in caught.value.detail
    with FakeSuperset() as fake:
        # An unrelated failure is not read as a missing extension, and costs no probe.
        fake.reply(TOOL_PREFIX + "editors_of", lambda request, authorization: RuntimeError("boom"))
        with pytest.raises(Exception) as caught:
            call_tool(_client(fake), "editors_of", {})
        assert not isinstance(caught.value, ExtensionMissing)
        assert [c.tool for c in fake.calls] == [TOOL_PREFIX + "editors_of"]


def test_require_extension_asks_once():
    """Whether Superset offers the extension, settled by one call of the probe: the installed extension answers it with
    an empty list; a Superset without the extension answers rc2's masked error, which reads as the extension missing;
    any other failure is raised as it is. Never a second call."""
    from mnemiq.superset.client import SupersetToolError
    from mnemiq.superset.extension import require_extension

    with FakeSuperset() as fake:
        fake.reply(DATASET_METRICS, lambda request, authorization: {"datasets": []})
        assert require_extension(_client(fake)) is None
        assert [(c.tool, c.request) for c in fake.calls] == [(DATASET_METRICS, {"dataset_ids": []})]
    with FakeSuperset(extension=False) as fake:
        with pytest.raises(ExtensionMissing) as caught:
            require_extension(_client(fake))
        assert "Error ID" not in str(caught.value) and "Error ID" in caught.value.detail
        assert [c.tool for c in fake.calls] == [DATASET_METRICS]
    with FakeSuperset() as fake:
        fake.reply(DATASET_METRICS, lambda request, authorization: RuntimeError("boom"))
        with pytest.raises(SupersetToolError) as caught:
            require_extension(_client(fake))
        assert not isinstance(caught.value, ExtensionMissing)
        assert [c.tool for c in fake.calls] == [DATASET_METRICS]


def test_require_build_tools_tells_an_older_extension_from_none():
    """Whether Superset offers the creates a build sends, settled by one call of the extension's `create_dataset` with
    an empty request, which this extension refuses without touching Superset. rc2's masked error is then told apart
    by the extension's own probe: an extension that answers it is older than this mnemiq (`ExtensionOutdated`); one
    that does not is missing. An answer that is not a refusal, and any other failure, are raised as tool errors."""
    from mnemiq.superset.client import SupersetToolError
    from mnemiq.superset.extension import ExtensionOutdated, require_build_tools

    create = TOOL_PREFIX + "create_dataset"
    masked = RuntimeError(f"Internal error in {create}: An unexpected error occurred. Error ID: 2.")
    with FakeSuperset() as fake:
        fake.reply(create, lambda request, authorization: {"refused": "request", "field": "database_id"})
        assert require_build_tools(_client(fake)) is None
        assert [(c.tool, c.request) for c in fake.calls] == [(create, {})]
    with FakeSuperset(extension=False) as fake:
        with pytest.raises(ExtensionMissing) as caught:
            require_build_tools(_client(fake))
        assert not isinstance(caught.value, ExtensionOutdated)
        assert [c.tool for c in fake.calls] == [create, DATASET_METRICS]
    with FakeSuperset() as fake:
        fake.reply(create, lambda request, authorization: masked)
        fake.reply(DATASET_METRICS, lambda request, authorization: {"datasets": []})
        with pytest.raises(ExtensionOutdated) as caught:
            require_build_tools(_client(fake))
        assert "older than this mnemiq" in str(caught.value) and "Error ID" not in str(caught.value)
        assert "Error ID" in caught.value.detail
        assert [c.tool for c in fake.calls] == [create, DATASET_METRICS]
    for answer in ({"dataset_id": 5}, RuntimeError("boom")):
        with FakeSuperset() as fake:
            fake.reply(create, lambda request, authorization, answer=answer: answer)
            with pytest.raises(SupersetToolError):
                require_build_tools(_client(fake))
            assert [c.tool for c in fake.calls] == [create]


def test_equivalent_saved_metric_is_reused():
    saved = [_saved("total_amount", "SUM(amount)"), _saved("other", "SUM(amount)", dataset_id=2, certified_by="x")]
    assert normalized("sum( amount )") == normalized("SUM(amount)")
    hit = reusable(_measure("sum( amount )"), saved)
    assert hit is not None and (hit.dataset_id, hit.name) == (1, "total_amount")
    assert reusable(_measure("sum( amount )", dataset_id=3), saved) is None
    assert reusable(_measure("avg(amount)"), saved) is None
    # What does not parse is never equal to anything, itself included.
    assert reusable(_measure("sum(("), [_saved("bad", "sum((")]) is None


def test_certified_metric_wins_a_tie():
    saved = [_saved("a_plain", "SUM(amount)"), _saved("b_certified", "SUM( amount )", certified_by="finance"),
             _saved("c_certified", "SUM(amount)", certified_by="finance")]
    assert reusable(_measure("SUM(amount)"), saved).name == "b_certified"


def test_over_limit_answer_is_a_stated_failure(fake):
    sizes: list[int] = []

    def too_big_once(request, authorization):
        sizes.append(request["page_size"])
        if request["page_size"] == 25:
            return RuntimeError(TOO_LARGE)
        return {"metrics": [_metric(1, "a")], "total_count": 1, "page": 1, "page_size": 5, "total_pages": 1,
                "success": True}

    fake.reply("list_metrics", too_big_once)
    _serve_extension(fake, {})
    assert [s.name for s in read_saved_metrics(_client(fake), dataset_ids=(1,))] == ["a"]
    assert sizes == [25, 5]

    sizes.clear()
    fake.reply("list_metrics", lambda request, authorization: (sizes.append(request["page_size"]),
                                                               RuntimeError(TOO_LARGE))[1])
    with pytest.raises(SupersetResponseTooLarge):
        read_saved_metrics(_client(fake), dataset_ids=(1,))
    assert sizes == [25, 5]


def test_a_metric_the_extension_does_not_answer_for_is_unknown_not_none(fake):
    _serve_metrics(fake, {1: [_metric(1, "a"), _metric(1, "b"), _metric(1, "c")], 2: [_metric(2, "z")]})
    # Dataset 2 is absent from the answer, and so is metric "b" of dataset 1; "c" is answered with null fields.
    _serve_extension(fake, {1: [_extension_row("a", currency=USD, certified_by="finance"), _extension_row("c")]})
    by_name = {(s.dataset_id, s.name): s for s in read_saved_metrics(_client(fake), dataset_ids=(1, 2))}
    for key in ((1, "b"), (2, "z")):
        assert by_name[key].currency is UNKNOWN and by_name[key].certified_by is UNKNOWN
    assert by_name[(1, "c")].currency is None and by_name[(1, "c")].certified_by is None
    assert by_name[(1, "a")].currency == USD and by_name[(1, "a")].certified_by == "finance"


def test_a_dataset_the_extension_marked_too_large_is_unknown_not_none(fake):
    """The extension answers a dataset whose metrics do not fit its size cap as `{dataset_id, too_large}` with no
    `metrics`: its saved metrics are unknown (the builder saw them in `list_metrics`), never "without currency"."""
    _serve_metrics(fake, {1: [_metric(1, "a")], 2: [_metric(2, "z")]})

    def answer(request, authorization):
        return {"datasets": [{"dataset_id": 1, "metrics": [_extension_row("a", currency=USD)]},
                             {"dataset_id": 2, "too_large": True}]}

    fake.reply(DATASET_METRICS, answer)
    by_name = {(s.dataset_id, s.name): s for s in read_saved_metrics(_client(fake), dataset_ids=(1, 2))}
    assert by_name[(2, "z")].currency is UNKNOWN and by_name[(2, "z")].certified_by is UNKNOWN
    assert by_name[(1, "a")].currency == USD


# ---------------------------------------------------------------- carried onto a join

# claim, joined to policy as `c`: policy_id and created_at are in both tables, so the joined dataset names claim's
# `claim_policy_id` and `claim_created_at`, and the rest as they are.
CLAIM = frozenset({"claim_id", "policy_id", "amount", "status", "created_at"})
CLAIM_SHOWN = {"claim_id": "claim_id", "policy_id": "claim_policy_id", "amount": "amount", "status": "status",
               "created_at": "claim_created_at"}
NOT_ONE = "mnemiq cannot read its expression as one SQL expression"
OTHER_ROWS = "it reads more than its own table's rows: a subquery, EXISTS or another table"
NO_COLUMN = "it names no column of claim, so on a join it would count the join's rows"


def _carry(expression: str, *, table: str = "claim", stored: str = "claim", columns=CLAIM,
           qualifier: exp.Identifier | None = None, shown=None, **more):
    return carry(_saved("m", expression), table=table, stored=stored, columns=columns,
                 qualifier=exp.to_identifier("c") if qualifier is None else qualifier,
                 shown=CLAIM_SHOWN if shown is None else shown, **more)


@pytest.mark.parametrize("expression, decided, superset", [
    ("SUM(amount)", 'SUM(c."amount")', 'SUM("amount")'),
    # Written with its table's name, which the join knows by its alias.
    ("SUM(claim.amount)", 'SUM(c."amount")', 'SUM("amount")'),
    ("SUM(Claim.Amount)", 'SUM(c."amount")', 'SUM("amount")'),
    ('SUM("claim"."amount")', 'SUM(c."amount")', 'SUM("amount")'),
    # A column the joined tables share is the joined dataset's `<table>_<column>`.
    ("MAX(created_at)", 'MAX(c."created_at")', 'MAX("claim_created_at")'),
    ("COUNT(DISTINCT policy_id)", 'COUNT(DISTINCT c."policy_id")', 'COUNT(DISTINCT "claim_policy_id")'),
    ("SUM(amount) FILTER (WHERE status = 'open')",
     'SUM(c."amount") FILTER(WHERE c."status" = \'open\')', 'SUM("amount") FILTER(WHERE "status" = \'open\')'),
    ("SUM(amount::numeric)", 'SUM(CAST(c."amount" AS DECIMAL))', 'SUM(CAST("amount" AS DECIMAL))'),
    ("COUNT(CASE WHEN created_at >= TIMESTAMPTZ '2024-01-01 00:00:00+00' THEN claim_id END)",
     'COUNT(CASE WHEN c."created_at" >= CAST(\'2024-01-01 00:00:00+00\' AS TIMESTAMPTZ) THEN c."claim_id" END)',
     'COUNT(CASE WHEN "claim_created_at" >= CAST(\'2024-01-01 00:00:00+00\' AS TIMESTAMPTZ) THEN "claim_id" END)'),
    ("SUM(CASE WHEN created_at > CURRENT_DATE - INTERVAL '30 days' THEN amount ELSE 0 END)",
     'SUM(CASE WHEN c."created_at" > CURRENT_DATE - INTERVAL \'30 DAYS\' THEN c."amount" ELSE 0 END)',
     'SUM(CASE WHEN "claim_created_at" > CURRENT_DATE - INTERVAL \'30 DAYS\' THEN "amount" ELSE 0 END)'),
])
def test_a_carried_metric_names_each_column_as_the_join_and_as_the_joined_dataset(expression, decided, superset):
    """One parse, printed twice: the decided rendering qualifies each column by the table's join qualifier, the
    Superset rendering names it as the joined dataset does; they differ only in column references."""
    assert _carry(expression) == Carried(decided=decided, superset=superset)


def test_a_quoted_table_is_qualified_with_its_quoting():
    lines = frozenset({"line_id", "policy_id", "amount"})
    shown = {"line_id": "line_id", "policy_id": "Policy Lines_policy_id", "amount": "Policy Lines_amount"}
    carried = _carry('SUM("Policy Lines".amount) FILTER (WHERE line_id > 0)', table="Policy Lines",
                     stored="Policy Lines", columns=lines, qualifier=exp.to_identifier("Policy Lines", quoted=True),
                     shown=shown)
    assert carried == Carried(decided='SUM("Policy Lines"."amount") FILTER(WHERE "Policy Lines"."line_id" > 0)',
                              superset='SUM("Policy Lines_amount") FILTER(WHERE "line_id" > 0)')


def test_a_column_may_be_qualified_by_the_table_as_superset_stores_it():
    """A table the snapshot names `party` and Superset stores as `Party`: a metric of its dataset may name either."""
    party = frozenset({"party_id", "policy_id"})
    shown = {"party_id": "party_id", "policy_id": "party_policy_id"}
    for written in ('COUNT("Party".party_id)', "COUNT(party.party_id)", 'COUNT("party".party_id)'):
        assert _carry(written, table="party", stored="Party", columns=party, shown=shown) \
            == Carried(decided='COUNT(c."party_id")', superset='COUNT("party_id")'), written
    assert _carry('COUNT("PARTY".party_id)', table="party", stored="Party", columns=party, shown=shown) \
        == 'it names "PARTY".party_id, a column of another table'


@pytest.mark.parametrize("expression, reason", [
    ("SUM({{ amount }})", "it holds {{, which Superset reads as a template marker"),
    ("SUM(amount) {% if 1 %}{% endif %}", "it holds {%, which Superset reads as a template marker"),
    ("SUM(amount) {# note #}", "it holds {#, which Superset reads as a template marker"),
    ("SUM((", NOT_ONE),
    ("", NOT_ONE),
    ("SUM(amount), SUM(claim_id)", NOT_ONE),
    ("SUM(amount) FROM claim", NOT_ONE),
    ("DISTINCT amount", NOT_ONE),
    ("SUM(amount); SELECT 1", NOT_ONE),
    ("SUM(amount) AS total", NOT_ONE),
    ("SUM(amount) AS (a, b)", NOT_ONE),
    ("SUM(amount) + (SELECT MAX(premium) FROM policy)", OTHER_ROWS),
    ("CASE WHEN EXISTS (SELECT 1 FROM policy) THEN SUM(amount) END", OTHER_ROWS),
    ("SUM(amount) FILTER (WHERE policy_id IN (SELECT policy_id FROM policy))", OTHER_ROWS),
    ("SUM(amount) * (VALUES (2))", OTHER_ROWS),
    # sqlglot reads `(TABLE policy)` as a column named TABLE, aliased policy.
    ("SUM(amount) * (TABLE policy)", OTHER_ROWS),
    ("COUNT(claim.*)", "it names *"),
    ("SUM(public.claim.amount)", "it names a column with a schema"),
    ("SUM(policy.premium)", "it names policy.premium, a column of another table"),
    # The join's alias is not the table's name: a saved metric is defined on the table alone.
    ("SUM(c.amount)", "it names c.amount, a column of another table"),
    ("SUM(premium)", "it names premium, which claim does not have or the builder cannot read"),
    ('SUM("Amount")', 'it names "Amount", which claim does not have or the builder cannot read'),
    ("COUNT(*)", NO_COLUMN),
    ("SUM(1)", NO_COLUMN),
    ("COUNT(1)", NO_COLUMN),
])
def test_a_metric_that_cannot_be_carried_is_refused_with_mnemiqs_reason(expression, reason):
    assert _carry(expression) == reason


@pytest.mark.parametrize("expression", ["COUNT(*)", "SUM(1)", "COUNT(1)"])
def test_a_metric_that_names_no_column_is_carried_where_the_joins_rows_are_its_tables(expression):
    """A join whose rows are claim's, each once (`rows_are_base`: claim LEFT JOIN far sides each shown to hold each
    value once), counts claim's rows wherever an expression names no column, as claim alone does: the metric is
    carried as written. Told nothing of the join's rows, or told they are not claim's, carry refuses it."""
    assert _carry(expression, rows_are_base=True) == Carried(decided=expression, superset=expression)
    assert _carry(expression, rows_are_base=False) == NO_COLUMN
    assert _carry(expression) == NO_COLUMN


@pytest.mark.parametrize("expression, reason", [
    ("COUNT({{ rows }})", "it holds {{, which Superset reads as a template marker"),
    ("COUNT(*), SUM(1)", NOT_ONE),
    ("COUNT(*) + (SELECT COUNT(*) FROM policy)", OTHER_ROWS),
    ("COUNT(claim.*)", "it names *"),
    ("COUNT(*) FILTER (WHERE policy.region = 'north')", "it names policy.region, a column of another table"),
    ("SUM(premium)", "it names premium, which claim does not have or the builder cannot read"),
])
def test_rows_that_are_the_tables_lift_only_the_refusal_of_a_metric_naming_no_column(expression, reason):
    assert _carry(expression, rows_are_base=True) == reason


def test_rows_that_are_the_tables_carry_a_metric_naming_columns_as_any_join_does():
    assert _carry("SUM(amount) FILTER (WHERE status = 'open')", rows_are_base=True) \
        == _carry("SUM(amount) FILTER (WHERE status = 'open')")


def test_a_column_the_builder_cannot_read_is_refused():
    assert _carry("SUM(amount)", columns=CLAIM - {"amount"}) \
        == "it names amount, which claim does not have or the builder cannot read"


_TABLES = {"policy": ("policy_id", "region", "premium", "start_date", "created_at"),
           "claim": ("claim_id", "policy_id", "amount", "status", "created_at"),
           "Policy Lines": ("line_id", "policy_id", "amount")}
_FILTERED = "SUM(amount) FILTER (WHERE created_at >= TIMESTAMPTZ '2024-01-01 00:00:00+00' AND policy_id > 0)"


@pytest.mark.parametrize("join, table, expression", [
    ("policy p JOIN claim c ON c.policy_id = p.policy_id", "claim", _FILTERED),
    ("policy JOIN claim ON claim.policy_id = policy.policy_id", "claim", _FILTERED),
    ('"Policy Lines" JOIN claim c ON c.policy_id = "Policy Lines".policy_id', "Policy Lines",
     'SUM("Policy Lines".amount) FILTER (WHERE line_id > 0 AND policy_id > 0)'),
])
def test_the_joined_dataset_reads_the_superset_rendering_as_the_decided_one(join, table, expression):
    """The joined dataset's SELECT, as the planner makes it, maps each column of the Superset rendering back to the
    joined column it projects; the result is the decided rendering, character for character."""
    from mnemiq.superset.planner import PlannedDataset, _Catalogue, _joined

    catalogue = _Catalogue(columns={t: frozenset(c) for t, c in _TABLES.items()}, stored={t: t for t in _TABLES},
                           datasets={}, saved={}, types={t: {} for t in _TABLES})
    rows = _joined(PlannedDataset(key="dataset_1", placeholder=-1, join=join), catalogue)
    assert not isinstance(rows, str), rows
    node = next(n for n in sqlglot.parse_one(f"SELECT 1 FROM {join}", read="postgres").find_all(exp.Table)
                if n.name == table)
    qualifier = node.args["alias"].this if node.args.get("alias") else node.this
    shown = {column: name for name, (owner, column) in rows.origins.items() if owner == table}
    carried = carry(_saved("m", expression), table=table, stored=table, columns=catalogue.columns[table],
                    qualifier=qualifier, shown=shown)
    assert isinstance(carried, Carried), carried
    assert carried.decided != carried.superset
    projected = {item.alias: item.this for item in sqlglot.parse_one(rows.planned.sql, read="postgres").expressions}
    back = sqlglot.parse_one(f"SELECT {carried.superset}", read="postgres").expressions[0].transform(
        lambda n: projected[n.name].copy() if isinstance(n, exp.Column) else n)
    assert back.sql(dialect="postgres") == carried.decided


# A table with columns named as Postgres's niladic functions: only quoted, or qualified, is such a name the column.
NILADIC = frozenset({"user", "current_role", "system_user", "amount"})
NILADIC_SHOWN = {name: name for name in NILADIC}


@pytest.mark.parametrize("expression, name", [
    ("MAX(user)", "user"), ("MAX(USER)", "USER"), ("MAX(current_role)", "current_role"),
    ("MAX(system_user)", "system_user"),
])
def test_a_name_postgres_reads_as_a_function_is_refused(expression, name):
    """sqlglot reads these unquoted names as columns, where Postgres reads the session's user or role: carried as
    `c."user"`, the metric would read the column instead."""
    assert _carry(expression, columns=NILADIC, shown=NILADIC_SHOWN) \
        == f"it names {name}, which Postgres reads as a function, not a column"


@pytest.mark.parametrize("expression", ['MAX("user")', "MAX(claim.user)"])
def test_a_quoted_or_qualified_niladic_name_is_the_column(expression):
    assert _carry(expression, columns=NILADIC, shown=NILADIC_SHOWN) \
        == Carried(decided='MAX(c."user")', superset='MAX("user")')


@pytest.mark.parametrize("expression", [
    # Each trips one check alone: a second SELECT (no subquery node), and EXISTS (over VALUES, no second SELECT).
    "SUM(amount) * ARRAY_LENGTH(ARRAY(SELECT 1), 1)",
    "CASE WHEN EXISTS (VALUES (1)) THEN SUM(amount) END",
    "CASE WHEN EXISTS (SELECT 1) THEN SUM(amount) END",
])
def test_a_second_select_or_exists_alone_is_refused(expression):
    assert _carry(expression) == OTHER_ROWS


def test_a_table_node_alone_is_refused(monkeypatch):
    """No Postgres text sqlglot parses to a table node without a subquery or a second SELECT around it (probed:
    `IN ROWS FROM (f())`, `ANY (TABLE t)`, `EXISTS (VALUES ...)`, ...), so the tree is built: a table node alone
    still refuses the metric."""
    import mnemiq.superset.saved_metrics as module

    tree = exp.select(exp.Mul(this=exp.Sum(this=exp.column("amount")), expression=exp.to_table("policy")))
    assert len(list(tree.find_all(exp.Select))) == 1 and not tree.find(exp.Subquery, exp.Exists, exp.Alias)
    monkeypatch.setattr(module.sqlglot, "parse_one", lambda *args, **kwargs: tree)
    assert _carry("SUM(amount)") == OTHER_ROWS


def test_the_qualifier_passed_in_is_left_as_it_was():
    qualifier = exp.to_identifier("c")
    assert _carry("SUM(amount) FILTER (WHERE status = 'open')", qualifier=qualifier) \
        == Carried(decided='SUM(c."amount") FILTER(WHERE c."status" = \'open\')',
                   superset='SUM("amount") FILTER(WHERE "status" = \'open\')')
    assert qualifier.parent is None and qualifier.sql(dialect="postgres") == "c"
