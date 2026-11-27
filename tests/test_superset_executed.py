"""`executed.check_executed`: on Oracle mnemiq sends a statement to Superset only as Superset would run it, read from
Superset's own dry run, and `SupersetSqlAdapter` makes that dry run before every real `execute_sql`.

What Superset would run is taken from rc2's own code: `tests/fixtures/superset_rc2_executed.json` was made by
`tests/fixtures/superset_rc2_executed.py` with Superset 7.0.0rc2 and its sqlglot, which the file records. The adapter's
tests drive it over a client that answers each tool as Superset and the extension would.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mnemiq.execute.runner import ExecutionError, SourceUnavailableError, run
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.client import SupersetRefusedCaller, SupersetToolError, SupersetUnavailable
from mnemiq.superset.executed import check_executed, check_tables_read, executed_sql, physical_reads, rule_key
from mnemiq.superset.extension import TOOL_PREFIX, ExtensionMissing, ExtensionOutdated
from mnemiq.superset.names import RefusedStatement
from mnemiq.superset.payload import SupersetAccessRefused, SupersetQueryFailed

RC2 = json.loads((Path(__file__).parent / "fixtures" / "superset_rc2_executed.json").read_text())
CASES = RC2["cases"]
TABLE_RULES = TOOL_PREFIX + "table_rules"
TABLES_READ = TOOL_PREFIX + "tables_read"


def _rules(case: dict, *, every: bool | None = None) -> dict[str, bool | None]:
    """The rules `check_executed` reads for a fixture's statement: whether a rule applies to each table it reads, as
    the case put rules on them (or `every` for all of them)."""
    ruled = set(case["ruled"])
    return {key: (name in ruled if every is None else every)
            for key, (_, _, name) in physical_reads(case["sent"], dialect="oracle").items()}


def _check(name: str, *, rules: dict | None = None, executed: str | None = None, limit: int | None = None) -> None:
    case = CASES[name]
    check_executed(case["sent"], case["executed"] if executed is None else executed,
                   rules=_rules(case) if rules is None else rules, dialect="oracle",
                   limit=case["limit"] if limit is None else limit)


def test_the_fixtures_were_made_by_superset_7_0_0rc2_and_its_sqlglot():
    """Remade with another Superset or sqlglot, the cases below say what changed."""
    assert (RC2["superset"], RC2["sqlglot"]) == ("7.0.0", "30.17.0")


@pytest.mark.parametrize("name", [
    "one table under its rule", "one table with no rule", "a join, one table ruled", "a CTE, a subquery and a schema",
    "a column qualified by its table's name", "a sampled table", "a table read twice", "a quoted table",
    "Superset's limit", "Superset's limit after an offset", "a subquery's own limit", "a statement's own limit",
    "the proof's probe",
])
def test_a_statement_superset_would_run_as_sent_passes(name):
    """rc2's own wrap of each table under its rule, and its limit in place of the statement's, read as the statement
    mnemiq sent."""
    _check(name)


def test_a_compared_is_not_null_as_mnemiq_prints_it_passes():
    """`(NOT x IS NULL) = 1`, as mnemiq prints a compared IS NOT NULL, rc2 renders as mnemiq reads it."""
    _check("IS NOT NULL compared, as mnemiq prints it")


@pytest.mark.parametrize("name", [
    "CROSS APPLY", "OUTER APPLY", "TO_DATE with its NLS argument", "TO_TIMESTAMP with its NLS argument",
    "GROUP BY a column named sample", "IS NOT NULL compared", "a subquery's outer-join mark",
])
def test_a_statement_superset_would_render_otherwise_is_refused(name):
    """rc2 renders each of these otherwise than mnemiq reads it: APPLY as a JOIN LATERAL, an NLS argument dropped, a
    column named sample read as a SAMPLE clause, `X IS NOT NULL = 1` as `NOT X IS NULL = 1`, a subquery's `(+)`
    dropped (mnemiq reads it, and prints the subquery without it)."""
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        _check(name)


def test_the_refusal_quotes_where_the_statement_was_read_otherwise():
    with pytest.raises(RefusedStatement) as refused:
        _check("TO_DATE with its NLS argument")
    assert "TO_DATE('01-JAN-2024', 'DD-MON-YYYY', 'NLS_DATE_LANGUAGE = AMERICAN')" in str(refused.value)
    assert "write that part of it another way" in str(refused.value)


def test_a_qualifier_two_tables_share_is_refused():
    """`SELECT ORDERS.ID FROM SHOP.ORDERS, ORDERS`: mnemiq refuses a statement whose qualifier, in the SQL Superset
    would run, names no one source of its query, with a rule on the tables or without."""
    for name in ("a qualifier two tables share", "a qualifier two tables share, no rule"):
        with pytest.raises(RefusedStatement, match=r"ORDERS\.ID"):
            _check(name)


def test_a_qualifier_naming_another_source_in_what_superset_would_run_is_refused():
    """mnemiq refuses a statement whose qualifier would resolve to another source in the SQL Superset would run than
    in the statement sent: `t.x` names the subquery's `t` as sent, and the outer `"T"` in what Superset would run."""
    with pytest.raises(RefusedStatement, match=r"t\.x"):
        _check("a qualifier that would name an outer table")


def test_a_bare_occurrence_of_a_ruled_table_is_refused():
    """A table a rule applies to, bare in the statement Superset would run, is refused."""
    bare = CASES["one table with no rule"]["executed"]
    with pytest.raises(RefusedStatement, match="orders"):
        _check("one table under its rule", executed=bare)


@pytest.mark.parametrize("told", [None, "absent"], ids=["cannot tell", "not asked"])
def test_a_bare_table_mnemiq_cannot_tell_the_rules_of_is_refused(told):
    """A table whose rules mnemiq cannot tell, or did not ask about, must be wrapped too."""
    with pytest.raises(RefusedStatement, match="orders"):
        _check("one table with no rule", rules={} if told == "absent" else {"orders": None})


def test_a_table_wrapped_with_no_rule_said_passes():
    """A table Superset wraps though `table_rules` said no rule applies is read under a rule, never more rows."""
    _check("one table under its rule", rules={"orders": False})


def test_one_occurrence_bare_of_a_table_read_twice_is_refused():
    """Every occurrence of a ruled table is wrapped, not only one."""
    executed = CASES["a table read twice"]["executed"]
    once = executed.replace('JOIN (\n  SELECT\n    *\n  FROM orders\n  WHERE\n    region = \'west\'\n) "B"',
                            'JOIN orders "B"')
    assert once != executed
    with pytest.raises(RefusedStatement, match="orders"):
        _check("a table read twice", executed=once)


@pytest.mark.parametrize(("old", "new"), [
    ("FROM orders\n  WHERE", "FROM customers\n  WHERE"),
    (') "O"', ') "C"'),
    ("SELECT\n    *\n  FROM orders", "SELECT\n    id\n  FROM orders"),
], ids=["another table", "another alias", "not every column"])
def test_a_wrap_that_is_not_rc2s_wrap_of_the_table_is_refused(old, new):
    executed = CASES["one table under its rule"]["executed"]
    assert old in executed
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        _check("one table under its rule", executed=executed.replace(old, new))


def test_a_cte_reference_superset_wraps_is_refused():
    """Only a table mnemiq reads may stand under a wrap: a CTE's reference wrapped is not the statement mnemiq read."""
    case = CASES["a CTE, a subquery and a schema"]
    executed = case["executed"].replace("FROM recent r", "FROM (SELECT * FROM recent WHERE region = 'west') r")
    assert executed != case["executed"]
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        _check("a CTE, a subquery and a schema", executed=executed)


def test_an_executed_statement_that_reads_differently_is_refused():
    executed = CASES["one table under its rule"]["executed"].replace("'open'", "'shut'")
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        _check("one table under its rule", executed=executed)


def test_an_executed_statement_selecting_fewer_columns_is_refused():
    executed = CASES["one table under its rule"]["executed"]
    assert "o.id,\n  o.status" in executed
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        _check("one table under its rule", executed=executed.replace("o.id,\n  o.status", "o.id"))


def test_only_the_top_level_limit_is_excepted():
    """A subquery's own row limit is the statement's, where a limit is asked or not."""
    executed = CASES["a subquery's own limit"]["executed"]
    assert "FETCH FIRST 5 ROWS ONLY" in executed
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        _check("a subquery's own limit", executed=executed.replace("FETCH FIRST 5 ROWS", "FETCH FIRST 50 ROWS"))


def test_supersets_limit_where_none_was_asked_is_refused():
    """The limit is excepted only where the call asked Superset for one."""
    case = CASES["Superset's limit"]
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        check_executed(case["sent"], case["executed"], rules=_rules(case), dialect="oracle", limit=None)


@pytest.mark.parametrize("asked", [6, None], ids=["fewer asked", "a limit of no count"])
def test_supersets_limit_must_be_at_most_the_rows_asked(asked):
    case = CASES["Superset's limit"]
    executed = case["executed"] if asked else case["executed"].replace("FETCH FIRST 7 ROWS ONLY",
                                                                       "FETCH FIRST 7 PERCENT ROWS ONLY")
    with pytest.raises(RefusedStatement, match="row limit"):
        check_executed(case["sent"], executed, rules=_rules(case), dialect="oracle", limit=asked or 7)


@pytest.mark.parametrize("executed", ["SELECT o.id FROM orders \"O\"; SELECT 1 FROM dual", "SELEC nothing FROM ("],
                         ids=["two statements", "unreadable"])
def test_an_executed_text_mnemiq_cannot_read_as_one_statement_is_refused(executed):
    with pytest.raises(RefusedStatement):
        _check("one table with no rule", executed=executed)


@pytest.mark.parametrize("text", ["EXPLAIN PLAN FOR SELECT * FROM secret", "CALL p()", "SHOW TABLES",
                                  "SELECT 1 FROM dual UNION ALL (EXPLAIN PLAN FOR SELECT 1 FROM dual)",
                                  'SELECT o.id FROM shop..orders "O"'])
def test_only_a_query_is_checked_on_either_side(text):
    """A text mnemiq's parser reads as no query, or holding a command it does not model, is unreadable sent or run:
    it reads no table for mnemiq, so nothing about it could be confirmed."""
    query = CASES["one table with no rule"]
    with pytest.raises(RefusedStatement, match="could not read"):
        check_executed(text, text, rules={}, dialect="oracle")
    with pytest.raises(RefusedStatement, match="mnemiq could not read the statement"):
        check_executed(text, query["executed"], rules={}, dialect="oracle")
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        check_executed(query["sent"], text, rules={"orders": False}, dialect="oracle")
    with pytest.raises(RefusedStatement, match="mnemiq could not read the statement"):
        physical_reads(text, dialect="oracle")


def test_a_sent_text_mnemiq_cannot_read_as_one_statement_is_refused():
    with pytest.raises(RefusedStatement, match="mnemiq could not read the statement as one query"):
        check_executed("SELEC nothing FROM (", CASES["one table with no rule"]["executed"], rules={}, dialect="oracle")


def test_a_dry_runs_one_executed_statement_is_read():
    payload = {"success": True, "statements": [{"original_sql": "x", "executed_sql": "SELECT 1 FROM dual"}]}
    assert executed_sql(payload) == "SELECT 1 FROM dual"


@pytest.mark.parametrize("statements", [None, [], "SELECT 1", [{"original_sql": "x"}], [{"executed_sql": 7}],
                                        [{"executed_sql": "SELECT 1 FROM dual"}] * 2])
def test_a_dry_run_without_exactly_one_executed_statement_is_refused(statements):
    with pytest.raises(RefusedStatement, match="dry run"):
        executed_sql({"success": True, "statements": statements})


def test_the_tables_read_are_keyed_as_supersets_rule_lookup_names_them():
    """Each table mnemiq reads, by the parts the statement names it with, a CTE's reference left out."""
    read = physical_reads(CASES["a CTE, a subquery and a schema"]["sent"], dialect="oracle")
    assert read == {"shop.orders": ("", "shop", "orders"), "lines": ("", "", "lines")}
    assert rule_key("", "", "OrderItems") == "OrderItems"
    assert rule_key("", "a.b", "c") == "a%2Eb.c"


# ---------------------------------------------------------------- the tables Superset's access check reads

SENT = 'SELECT o.id FROM orders "O" JOIN shop2.lines "L" ON l.order_id = o.id'


def _read(*tables) -> frozenset:
    return frozenset(tables)


def test_every_table_mnemiq_reads_must_be_one_superset_reads():
    """Each table mnemiq reads, by the schema that qualifies it or the session's, must be among the tables Superset's
    access check reads (the extension's `tables_read`); a table only Superset reads asks nothing of mnemiq, and the
    catalog Superset qualifies an unqualified table by is the database's own."""
    both = _read(("live", "shop", "orders"), ("live", "shop2", "lines"))
    check_tables_read(SENT, both, schema="shop", dialect="oracle")
    check_tables_read(SENT, both | {("live", "shop", "extra")}, schema="shop", dialect="oracle")
    check_tables_read(SENT, _read((None, "shop", "orders"), (None, "shop2", "lines")), schema="shop", dialect="oracle")
    check_tables_read('WITH r AS (SELECT o.id FROM orders "O") SELECT r.id FROM r', _read((None, "shop", "orders")),
                      schema="shop", dialect="oracle")


@pytest.mark.parametrize(("read", "named"), [
    (_read(("live", "shop", "orders")), "shop2.lines"),
    (_read(("live", "shop", "orders"), ("live", "shop", "lines")), "shop2.lines"),
    (_read(("live", "shop2", "orders"), ("live", "shop2", "lines")), "orders"),
    (_read(("live", "shop", "ORDERS"), ("live", "shop2", "lines")), "orders"),
], ids=["left out", "in another schema", "unqualified, in another schema", "in another case"])
def test_a_table_mnemiq_reads_and_superset_does_not_is_refused(read, named):
    with pytest.raises(RefusedStatement, match=f"which mnemiq could not confirm for {re.escape(named)}, so it was not"):
        check_tables_read(SENT, read, schema="shop", dialect="oracle")


def test_a_table_named_by_a_catalog_is_compared_by_it():
    sent = 'SELECT o.id FROM cat1.shop.orders "O"'
    check_tables_read(sent, _read(("cat1", "shop", "orders")), schema="shop", dialect="oracle")
    with pytest.raises(RefusedStatement, match="could not confirm"):
        check_tables_read(sent, _read(("cat2", "shop", "orders")), schema="shop", dialect="oracle")


def test_tables_superset_reads_that_mnemiq_cannot_tell_is_a_refusal():
    with pytest.raises(RefusedStatement, match="could not tell which tables Superset reads"):
        check_tables_read(SENT, None, schema="shop", dialect="oracle")


# ---------------------------------------------------------------- the adapter


ROWS = {"success": True, "rows": [{"id": 1}], "columns": [{"name": "id", "type": "int64"}], "row_count": 1}


class _Superset:
    """Stands in for `SupersetMcp` before Superset and the extension: records each call; answers a dry run with the
    statement it is given as `executed` (by default, a statement whose text Superset would run as sent), `tables_read`
    with `read` (by default, the tables mnemiq reads in the statement, in the schema asked where it names none),
    `table_rules` from `rules` by table name (`answers` in its place when given), and a run with one row."""

    def __init__(self, *, executed=None, rules: dict | None = None, answers=None, read=None, run=ROWS) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.executed, self.rules, self.answers, self.read, self.run = executed, rules or {}, answers, read, run

    def call(self, tool: str, request: dict, *, timeout_s: float | None = None) -> dict:
        self.calls.append((tool, request))
        if tool == "execute_sql" and request.get("dry_run"):
            executed = self.executed(request["sql"]) if callable(self.executed) else self.executed
            if isinstance(executed, (Exception, dict)):
                if isinstance(executed, Exception):
                    raise executed
                return executed
            return {"success": True, "rows": None, "columns": None,
                    "statements": [{"original_sql": request["sql"], "executed_sql": executed or request["sql"],
                                    "row_count": 0, "data": None}]}
        if tool == "execute_sql":
            return self.run
        if tool == TABLES_READ:
            if "sql" not in request:
                return {"refused": "request", "field": "sql"}  # the extension's answer to its probe
            answer = self.read(request) if callable(self.read) else self.read
            if isinstance(answer, Exception):
                raise answer
            if answer is not None:
                return answer
            return {"tables": [{"catalog": None, "schema": schema or request["schema"], "table": name}
                               for _, schema, name in physical_reads(request["sql"], dialect="oracle").values()]}
        if tool == TABLE_RULES:
            if "tables" not in request and not callable(self.answers):
                return {"refused": "the request names no tables"}  # the extension's answer to its probe
            if self.answers is not None:
                answer = self.answers(request) if callable(self.answers) else self.answers
                if isinstance(answer, Exception):
                    raise answer
                return answer
            return {"tables": [{"table": name, "applies": self.rules.get(name, False)} for name in request["tables"]]}
        raise AssertionError(f"unexpected call of {tool}")


def _oracle(superset: _Superset, **kwargs) -> SupersetSqlAdapter:
    return SupersetSqlAdapter(superset, database_id=7, schema="shop", dialect="oracle", **kwargs)


def _tools(superset: _Superset) -> list[str]:
    return [tool if not request.get("dry_run") else "dry run" for tool, request in superset.calls]


def test_on_oracle_the_dry_run_comes_first_and_the_real_call_after_the_check(superset_oracle):
    """The dry run, then `tables_read` of the statement sent, then `table_rules` for the tables it reads, then the
    real call, which carries the dry run's options but `dry_run`."""
    case = CASES["one table under its rule"]
    superset = _Superset(executed=case["executed"], rules={"orders": True})
    table = _oracle(superset).execute_arrow(case["sent"])
    assert table.num_rows == 1
    assert _tools(superset) == ["dry run", TABLES_READ, TABLE_RULES, "execute_sql"]
    (_, probe), (_, read), (_, asked), (_, real) = superset.calls
    assert probe == {**real, "dry_run": True} and "dry_run" not in real
    assert real == {"database_id": 7, "sql": case["sent"], "schema": "shop", "timeout": 30}
    assert read == {"database_id": 7, "schema": "shop", "sql": case["sent"]}
    assert asked == {"database_id": 7, "schema": "shop", "tables": ["orders"]}


def test_on_oracle_a_refused_statement_is_never_run(superset_oracle):
    """A bare occurrence of a ruled table: the check refuses it, as a statement for the model to repair, and the
    real call is never made."""
    case = CASES["one table under its rule"]
    superset = _Superset(executed=CASES["one table with no rule"]["executed"], rules={"orders": True})
    with pytest.raises(SupersetQueryFailed, match="orders"):
        _oracle(superset).execute_arrow(case["sent"])
    assert _tools(superset) == ["dry run", TABLES_READ, TABLE_RULES]


def test_on_oracle_supersets_own_refusal_of_the_dry_run_is_raised_as_from_a_run(superset_oracle):
    refused = {"success": False, "error": "You need access to the following tables: `shop.secret`",
               "error_type": "TABLE_SECURITY_ACCESS_ERROR"}
    superset = _Superset(executed=refused)
    with pytest.raises(Exception) as raised:
        _oracle(superset).execute_arrow('SELECT s.id FROM secret "S"')
    assert type(raised.value) is SupersetAccessRefused, raised.value
    assert _tools(superset) == ["dry run"]


def test_on_oracle_the_limit_and_refresh_the_real_call_carries_are_the_dry_runs(superset_oracle):
    """A statement asked again below its limit, by an adapter that refreshes every statement: the dry run carries that
    limit and that refresh, so Superset transforms the statement it then runs the same way, and the check excepts
    the limit Superset puts in place."""
    case = CASES["Superset's limit"]
    superset = _Superset(executed=case["executed"], rules={"orders": True})
    _oracle(superset, refresh_every_statement=True)._execute_sql(case["sent"], None, row_limit=7)
    (_, probe), _, _, (_, real) = superset.calls
    assert probe == {**real, "dry_run": True} and real["limit"] == 7 and real["force_refresh"] is True


def test_on_oracle_a_dry_run_asked_for_is_the_check_s_own(superset_oracle):
    """`dry_run` sends one dry run, checked, and no real call."""
    superset = _Superset()
    _oracle(superset).dry_run('SELECT o.id FROM orders "O"')
    assert _tools(superset) == ["dry run", TABLES_READ, TABLE_RULES]


def test_on_oracle_the_proofs_probe_is_checked_too(superset_oracle):
    case = CASES["the proof's probe"]
    superset = _Superset(executed=case["executed"], rules={"orders": True})
    adapter = _oracle(superset)
    superset.run = {"success": True, "rows": [], "columns": [{"name": "region", "type": "object"},
                                                             {"name": "n", "type": "int64"}]}
    adapter.validate('SELECT o.region, COUNT(*) AS n FROM orders "O" GROUP BY o.region')
    assert _tools(superset) == ["dry run", TABLES_READ, TABLE_RULES, "execute_sql"]
    assert superset.calls[0][1]["sql"] == case["sent"]


def test_on_oracle_tables_are_asked_by_the_schema_that_qualifies_them(superset_oracle):
    """One `table_rules` call per schema: a table the statement qualifies, in its schema; an unqualified one, in the
    session's."""
    superset = _Superset()
    _oracle(superset).execute_arrow('SELECT o.id FROM shop2.orders "O" JOIN lines "L" ON l.order_id = o.id '
                                    'JOIN shop2.items "I" ON i.id = l.item_id')
    asked = [request for tool, request in superset.calls if tool == TABLE_RULES]
    assert asked == [{"database_id": 7, "schema": "shop2", "tables": ["orders", "items"]},
                     {"database_id": 7, "schema": "shop", "tables": ["lines"]}]


def test_on_oracle_a_rule_superset_cannot_tell_of_is_one_the_table_must_be_wrapped_for(superset_oracle):
    """`table_rules` that cannot tell (a malformed answer) of an extension that offers it: the table must be wrapped,
    and bare it is refused."""
    superset = _Superset(answers={"tables": "not a list"})
    with pytest.raises(SupersetQueryFailed, match="could not tell whether one applies to orders"):
        _oracle(superset).execute_arrow('SELECT o.id FROM orders "O"')
    assert _tools(superset) == ["dry run", TABLES_READ, TABLE_RULES, TABLE_RULES]


@pytest.mark.parametrize(("missing", "raised"), [("all", ExtensionMissing), ("table_rules", ExtensionOutdated)])
def test_on_oracle_superset_without_table_rules_ends_the_request(superset_oracle, missing, raised):
    """Without the extension's `table_rules` (no extension, or one older than 0.3.0) nothing is sent: the adapter
    raises the extension's refusal, remembered as the request's end, and the runner reads it as a source that did
    not answer rather than a statement to repair."""
    masked = SupersetToolError("Error: Internal error in extensions.agenticfabriq.mnemiq.table_rules: An unexpected "
                               "error occurred. Error ID: x.",
                               detail="Error: Internal error in extensions.agenticfabriq.mnemiq.table_rules: An "
                                      "unexpected error occurred. Error ID: x.")

    def answers(request):
        return masked

    class _Without(_Superset):
        def call(self, tool, request, *, timeout_s=None):
            if tool == TOOL_PREFIX + "dataset_metrics":
                self.calls.append((tool, request))
                if missing == "all":
                    raise SupersetToolError("masked", detail="Error: Internal error in extensions.agenticfabriq."
                                                             "mnemiq.dataset_metrics: An unexpected error occurred.")
                return {"datasets": []}
            return super().call(tool, request, timeout_s=timeout_s)

    superset = _Without(answers=answers)
    adapter = _oracle(superset)
    with pytest.raises(raised):
        adapter.execute_arrow('SELECT o.id FROM orders "O"')
    assert isinstance(adapter.terminal_refusal, raised)
    assert "execute_sql" not in [tool for tool, request in superset.calls if not request.get("dry_run")]
    with pytest.raises(ExecutionError) as ended:
        run(_oracle(_Without(answers=answers)), 'SELECT o.id FROM orders "O"')
    assert isinstance(ended.value, SourceUnavailableError), ended.value


def test_on_oracle_a_refusal_of_the_caller_at_table_rules_is_remembered(superset_oracle):
    refused = SupersetRefusedCaller("refused")
    superset = _Superset(answers=lambda request: refused)
    adapter = _oracle(superset)
    with pytest.raises(SupersetRefusedCaller):
        adapter.execute_arrow('SELECT o.id FROM orders "O"')
    assert adapter.terminal_refusal is refused


def test_on_oracle_a_table_superset_does_not_read_as_mnemiq_does_is_never_sent(superset_oracle):
    """`tables_read` without a table mnemiq reads: nothing is sent, the reason naming the table, and `table_rules`
    is not asked."""
    superset = _Superset(read={"tables": [{"catalog": None, "schema": "shop", "table": "orders"}]})
    with pytest.raises(SupersetQueryFailed, match="could not confirm for lines"):
        _oracle(superset).execute_arrow('SELECT o.id FROM orders "O" JOIN lines "L" ON l.order_id = o.id')
    assert _tools(superset) == ["dry run", TABLES_READ]


def test_on_oracle_tables_read_that_cannot_tell_is_a_refusal(superset_oracle):
    """An answer mnemiq cannot read (here a text Superset says it cannot read) from an extension that offers the tool:
    refused, for the model to repair, after the probe that tells it from an older extension."""
    superset = _Superset(read={"unreadable": True})
    with pytest.raises(SupersetQueryFailed, match="could not tell which tables Superset reads"):
        _oracle(superset).execute_arrow('SELECT o.id FROM orders "O"')
    assert _tools(superset) == ["dry run", TABLES_READ, TABLES_READ]


@pytest.mark.parametrize("at", [TABLES_READ, TABLE_RULES])
def test_on_oracle_superset_not_answering_the_check_is_superset_unavailable(superset_oracle, at):
    """Superset not answering `tables_read` or `table_rules` is no answer mnemiq cannot read: the adapter raises it as
    Superset not answering, the runner ends the turn on it rather than repair the statement, and nothing is sent. The
    extension's probe is answered here, as Superset answers a call after one it did not."""
    silent = SupersetUnavailable("Superset did not answer")

    def rules(request):
        return silent if "tables" in request else {"refused": "request", "field": "tables"}

    superset = _Superset(read=silent if at == TABLES_READ else None, answers=rules if at == TABLE_RULES else None)
    with pytest.raises(SupersetUnavailable):
        _oracle(superset).execute_arrow('SELECT o.id FROM orders "O"')
    assert "execute_sql" not in [tool for tool, request in superset.calls if not request.get("dry_run")]
    with pytest.raises(ExecutionError) as ended:
        run(_oracle(superset), 'SELECT o.id FROM orders "O"')
    assert isinstance(ended.value, SourceUnavailableError), ended.value


def test_on_postgres_one_call_and_the_request_as_before():
    """PostgreSQL makes no dry run and no `table_rules` call, and sends what it always sent."""
    superset = _Superset()
    adapter = SupersetSqlAdapter(superset, database_id=7, schema="public")
    adapter.execute_arrow("SELECT id FROM orders")
    adapter.dry_run("SELECT id FROM orders")
    assert superset.calls == [("execute_sql", {"database_id": 7, "sql": "SELECT id FROM orders", "schema": "public",
                                               "timeout": 30}),
                              ("execute_sql", {"database_id": 7, "sql": "SELECT id FROM orders", "schema": "public",
                                               "timeout": 30, "dry_run": True})]


def test_confirm_sent_checks_a_statement_as_superset_would_run_it_on_oracle_only(superset_oracle):
    """The builder's and the chart check's door: Superset's dry run of a statement already in Superset's spelling,
    checked, and no real call; on PostgreSQL nothing at all."""
    case = CASES["CROSS APPLY"]
    superset = _Superset(executed=case["executed"])
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        _oracle(superset).confirm_sent(case["sent"])
    assert _tools(superset) == ["dry run", TABLES_READ, TABLE_RULES]
    called: list[str] = []
    _oracle(_Superset()).confirm_sent('SELECT o.id FROM orders "O"', before_call=lambda: called.append("renewed"))
    assert called == ["renewed"]
    postgres = _Superset()
    SupersetSqlAdapter(postgres, database_id=7, schema="public", dialect="postgres").confirm_sent(
        "SELECT id FROM orders", before_call=lambda: called.append("renewed"))
    assert postgres.calls == [] and called == ["renewed"]
