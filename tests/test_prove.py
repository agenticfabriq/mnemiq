"""The proof step the deciders run before approving a plan, and how it chooses its form.

`EXPLAIN <sql>` is Postgres and DuckDB syntax. Oracle raises ORA-02000 on it, so every Oracle plan
was refused before execution by the step whose purpose is to tell a valid plan from an invalid one.
Nothing caught it: the Oracle adapter's tests exercise the adapter, and the deciders' tests use
adapters that speak EXPLAIN. The failure lived in the gap between two well-tested things.
"""

from __future__ import annotations

from mnemiq.sql.prove import prove
from mnemiq.sql.verdict import Refusal, RefusalCode


class _Explains:
    """An adapter with no `validate`: the shape every non-Oracle adapter has."""

    def __init__(self, fail: str | None = None):
        self.seen: list[str] = []
        self._fail = fail

    def execute(self, sql):
        self.seen.append(sql)
        if self._fail:
            raise RuntimeError(self._fail)
        return []


class _Validates(_Explains):
    """An adapter that knows a better proof than EXPLAIN."""

    def __init__(self, fail: str | None = None):
        super().__init__()
        self.validated: list[str] = []
        self._validate_fail = fail

    def validate(self, sql):
        self.validated.append(sql)
        if self._validate_fail:
            raise RuntimeError(self._validate_fail)


def test_an_adapter_without_validate_still_gets_explain():
    """Back-compat is the point: three adapters rely on this and none of them changed."""
    a = _Explains()
    assert prove(a, "SELECT 1") is None
    assert a.seen == ["EXPLAIN SELECT 1"]


def test_an_adapter_with_validate_is_never_sent_explain():
    a = _Validates()
    assert prove(a, "SELECT 1") is None
    assert a.validated == ["SELECT 1"]
    assert a.seen == [], "EXPLAIN must not also be issued -- on Oracle it is a syntax error"


def test_a_rejection_becomes_explain_failed_carrying_the_sources_own_message():
    """The message is not decoration: the correction loop reads it to repair the SQL."""
    r = prove(_Explains(fail='relation "ghost" does not exist'), "SELECT 1 FROM ghost")
    assert isinstance(r, Refusal) and r.code is RefusalCode.EXPLAIN_FAILED
    assert 'relation "ghost" does not exist' in r.message


def test_a_validate_rejection_takes_the_same_path():
    r = prove(_Validates(fail="ORA-00942: table or view does not exist"), "SELECT 1 FROM ghost")
    assert isinstance(r, Refusal) and r.code is RefusalCode.EXPLAIN_FAILED
    assert "ORA-00942" in r.message


def test_the_seam_is_optional_rather_than_required():
    """A required method would break every test double that stands in for an adapter, and the
    breakage would surface as a source error rather than as a missing method."""

    class _Bare:
        def execute(self, sql):
            return []

    assert prove(_Bare(), "SELECT 1") is None
