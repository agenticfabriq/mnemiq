"""M157: Oracle's one-argument TRUNC is a builtin that reads nothing past its argument.

sqlglot leaves `trunc(d)` untyped in the oracle dialect, and lineage names every untyped call
outside `_PURE` as an object it cannot account for, so an Oracle answer truncating a date read
`unknown` with `unresolved=['trunc']` where the same answer with `round` read clean. The owner's
decision (2026-10-15): add TRUNC, under a standing rule for Oracle builtins (see `_PURE`).

What the entry must not do: clear a source's own `trunc`. `_PURE` only spares the name from the
record, and only where the inventory does not say the source may define it (Codex's review of
#157); whether the call is the builtin stays the function inventory's question.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.sql.decide import decide
from mnemiq.sql.functions import FunctionInventory
from mnemiq.sql.lineage import COMPLETE, UNKNOWN, lineage_for
from mnemiq.sql.verdict import Refusal, RefusalCode
from mnemiq.sql.views import ViewInventory


def _lineage(sql, functions):
    return lineage_for(sqlglot.parse_one(sql, read="oracle"), ["CLAIM"], ViewInventory(),
                       functions=functions)


def test_sqlglot_leaves_one_argument_trunc_untyped_in_oracle():
    """Why the entry is needed: a typed node never reaches the `_PURE` check at all."""
    call = sqlglot.parse_one("SELECT trunc(created) FROM CLAIM", read="oracle").selects[0]
    assert isinstance(call, exp.Anonymous), type(call)


def test_trunc_reads_as_a_typed_builtin_does_where_the_source_defines_no_functions():
    defines_none = FunctionInventory.of(())
    trunc = _lineage("SELECT trunc(created) AS d FROM CLAIM", defines_none)
    assert trunc.completeness == COMPLETE and trunc.unresolved == [], trunc
    assert trunc == _lineage("SELECT round(amount) AS d FROM CLAIM", defines_none)


class _OracleSource:
    """The Oracle door: asked by name, matching names case-insensitively as the adapter does."""

    binder_prefers_builtins = True

    def __init__(self, defined=()):
        self.defined = set(defined)

    def resolve_called_functions(self, names):
        return sorted(self.defined & {n.lower() for n in names}), []

    def execute(self, sql):  # `prove`'s EXPLAIN
        return []


def test_the_decider_records_trunc_as_it_records_round():
    def unresolved(sql):
        verdict = decide(sql, {"CLAIM": {"id", "created", "amount"}}, adapter=_OracleSource(),
                         dialect="oracle", target="oracle")
        return verdict.lineage.unresolved

    assert unresolved("SELECT trunc(created) AS d FROM CLAIM") == []
    assert unresolved("SELECT trunc(created) AS d FROM CLAIM") == unresolved(
        "SELECT round(amount) AS d FROM CLAIM")


def test_a_source_that_defines_trunc_still_cannot_confirm_the_call_and_the_record_names_it():
    """Codex's review of #157: the name is the suspect here, so sparing it lost the evidence."""
    lineage = _lineage("SELECT trunc(created) AS d FROM CLAIM", FunctionInventory.of({"trunc"}))
    assert lineage.completeness == UNKNOWN
    assert "unconfirmed-function-identity" in lineage.reasons
    assert lineage.unresolved == ["trunc"]


def test_a_narrow_answer_spares_trunc_only_when_it_was_asked_about_it():
    def unresolved(asked):
        inventory = FunctionInventory.of((), asked_names=asked, binder_prefers_builtins=True)
        return lineage_for(sqlglot.parse_one("SELECT trunc(created) AS d FROM CLAIM",
                                             read="oracle"), ["CLAIM"], ViewInventory(),
                           functions=inventory, called=["trunc"]).unresolved

    assert unresolved({"trunc"}) == []
    assert unresolved({"upper"}) == ["trunc"], "never asked: nothing rules a UDF out"


def test_an_older_entry_is_named_where_the_source_defines_it_too():
    """`now` was on the list before TRUNC, and had the same gap."""
    sql = sqlglot.parse_one("SELECT now() AS t FROM CLAIM", read="duckdb")
    assert isinstance(sql.selects[0].this, exp.Anonymous)  # untyped in duckdb, so it is listed
    named = lineage_for(sql, ["CLAIM"], ViewInventory(), functions=FunctionInventory.of({"now"}))
    spared = lineage_for(sql, ["CLAIM"], ViewInventory(), functions=FunctionInventory.of(()))
    assert named.unresolved == ["now"] and spared.unresolved == []


def test_a_view_body_s_trunc_is_named_where_the_source_defines_it():
    from mnemiq.contract import ViewDefinition

    views = ViewInventory({"v": ViewDefinition(object_id="v", dialect="oracle",
                                               definition="SELECT trunc(created) AS d FROM CLAIM")})
    lineage = lineage_for(sqlglot.parse_one("SELECT d FROM v", read="oracle"), ["v"], views,
                          functions=FunctionInventory.of({"trunc"}))
    assert "trunc" in lineage.unresolved, lineage


def test_a_qualified_trunc_is_not_the_builtin():
    assert "app.trunc" in _lineage("SELECT app.trunc(created) AS d FROM CLAIM",
                                   FunctionInventory.of(())).unresolved


def test_a_quoted_trunc_is_left_to_the_question_put_to_the_source():
    """sqlglot drops the quotes from a builtin-named call in Oracle, so `"trunc"(d)` reaches lineage
    as `trunc` and `_PURE` cannot tell it apart. The decider still asks the source about it, and a
    source defining `trunc` under any case refuses the statement before lineage is recorded."""
    verdict = decide('SELECT "trunc"(created) AS d FROM CLAIM', {"CLAIM": {"id", "created"}},
                     adapter=_OracleSource(defined={"trunc"}), dialect="oracle", target="oracle")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL, verdict
