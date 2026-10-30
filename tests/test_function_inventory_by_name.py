"""M101: an adapter whose binder prefers builtins is asked about the names a statement calls.

`decide` read an Oracle source's whole function inventory on every attempt, and the synonym
dictionary behind it costs its size however it is filtered: measured on Oracle Free 23.26.2 with
107,886 synonyms, 4.8 s a decision for `user_functions` and `unresolvable_aliases` together. The
owner's decision (2026-10-14): ask the narrower question -- which of THESE names are defined -- and
let lineage ask it too. Per-name is sound only where no binder renames a call, so only an adapter
declaring `binder_prefers_builtins` (Oracle, measured) is asked that way.

What a narrow answer must never do: clear a name it was not asked about. `answers_for` is how every
reader checks, and an unasked name refuses, stays unconfirmed, or counts as opaque.
"""

from __future__ import annotations

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.functions import FunctionInventory, calls_are_confirmable, inventory_from
from mnemiq.sql.verdict import Refusal, RefusalCode

VISIBLE = {"CLAIM": {"id", "name"}}


class _ByName:
    """An Oracle-shaped adapter: answers by name, binder prefers builtins, and RECORDS a whole read
    -- `read_whole` is how these tests know the cost went away. It does not raise there:
    `inventory_from` reads any exception as "could not answer", which would hide the read."""

    binder_prefers_builtins = True

    def __init__(self, defined=(), unresolvable=(), virtual=(), fail=False):
        self.defined, self.unresolvable, self.virtual, self.fail = (
            set(defined), set(unresolvable), list(virtual), fail)
        self.asked: list[frozenset] = []
        self.read_whole = False

    def resolve_called_functions(self, names):
        self.asked.append(frozenset(names))
        if self.fail:
            raise RuntimeError("ORA-03113")
        return sorted(self.defined & set(names)), sorted(self.unresolvable & set(names))

    def user_functions(self):
        self.read_whole = True
        return sorted(self.defined)

    def unresolvable_aliases(self):
        self.read_whole = True
        return sorted(self.unresolvable)

    def execute(self, sql):  # `prove`'s EXPLAIN, which a fake source accepts
        return []

    def virtual_columns(self):
        return self.virtual


def _decide(sql, adapter):
    return decide(sql, VISIBLE, adapter=adapter, dialect="oracle", target="oracle")


def test_a_builtin_call_is_decided_without_reading_the_whole_dictionary():
    adapter = _ByName(defined={"helper"})
    verdict = _decide("SELECT upper(name) FROM CLAIM", adapter)
    assert not isinstance(verdict, Refusal), verdict
    assert adapter.asked and "upper" in adapter.asked[0] and "helper" not in adapter.asked[0]
    assert not adapter.read_whole


def test_a_call_the_source_defines_is_refused():
    verdict = _decide("SELECT add_days(1) AS x FROM CLAIM", _ByName(defined={"add_days"}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL
    assert "defined by this source itself" in verdict.message


def test_a_call_through_a_link_alias_is_refused_with_its_own_sentence():
    verdict = _decide("SELECT add_days(1) AS x FROM CLAIM", _ByName(unresolvable={"add_days"}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL
    assert "synonym it cannot resolve" in verdict.message


def test_a_source_that_cannot_answer_refuses_the_statement():
    verdict = _decide("SELECT upper(name) FROM CLAIM", _ByName(fail=True))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_CALLS


def test_an_adapter_whose_binder_may_rename_is_still_asked_for_the_whole_source():
    """DuckDB binds `count(*)` to a macro named `count_star`, which no statement spells, so a
    list of spelled names is not the whole question there."""

    class _Renaming(_ByName):
        binder_prefers_builtins = False

    adapter = _Renaming()
    inventory = inventory_from(adapter, called={"upper"})
    assert adapter.read_whole and not adapter.asked and inventory.asked_names is None


def test_the_virtual_columns_calls_are_asked_about_too():
    """`opaque_columns` reads the same answer for them; unasked, a virtual column calling
    `upper` would be refused for want of a question."""
    adapter = _ByName(virtual=[("CLAIM", "loud", "UPPER(\"NAME\")")])
    inventory = inventory_from(adapter, called={"id"})
    assert {"id", "upper"} <= inventory.asked_names


NARROW = FunctionInventory.of({"helper"}, asked_names={"upper", "helper", "via_link"},
                              unresolvable={"via_link"}, binder_prefers_builtins=True)


@pytest.mark.parametrize("called,confirmable", [
    ({"upper"}, True),       # asked, and not the source's
    ({"helper"}, False),     # the source's own
    ({"via_link"}, False),   # through a link: a statement CALLING one cannot be confirmed
    ({"lower"}, False),      # never asked about: nothing rules it out
    (None, False),           # no names given at all
])
def test_a_narrow_answer_confirms_only_names_it_cleared(called, confirmable):
    assert calls_are_confirmable(NARROW, in_view_body=False, called=called) is confirmable


def test_a_name_never_asked_about_reads_as_defined():
    assert NARROW.defines("lower") and NARROW.defines("helper") and not NARROW.defines("upper")


def test_the_guard_refuses_a_call_the_answer_was_not_asked_about():
    """Two renderings disagreeing must not clear the call nobody asked about."""
    from mnemiq.sql.authz_guard import check_unmodelled_calls
    from sqlglot import parse_one

    asked_upper_only = FunctionInventory.of((), asked_names={"upper"}, binder_prefers_builtins=True)
    ast = parse_one("SELECT add_days(1) FROM CLAIM", read="oracle")
    assert check_unmodelled_calls(ast, asked_upper_only, "oracle", "oracle") is not None


def test_an_unrelated_function_no_longer_downgrades_the_answers_lineage():
    """The precision the owner chose with this: a source defining one helper used to mark every
    answer carrying a call `unconfirmed-function-identity` (issue #5's coarse rule)."""
    verdict = _decide("SELECT upper(name) FROM CLAIM", _ByName(defined={"helper"}))
    assert "unconfirmed-function-identity" not in verdict.lineage.reasons, verdict.lineage
    whole = FunctionInventory.of({"helper"}, binder_prefers_builtins=True)
    assert not calls_are_confirmable(whole, in_view_body=False)


def test_a_virtual_column_calling_a_name_never_asked_about_is_opaque():
    from mnemiq.sql.functions import opaque_columns

    adapter = _ByName(virtual=[("CLAIM", "loud", "UPPER(\"NAME\")"),
                               ("CLAIM", "leaked", "\"APP\".\"SECRET_FN\"(\"ID\")")])
    asked_upper_only = FunctionInventory.of((), asked_names={"upper"}, binder_prefers_builtins=True)
    assert opaque_columns(adapter, asked_upper_only) == frozenset({("CLAIM", "leaked")})
