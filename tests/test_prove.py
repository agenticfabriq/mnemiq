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
    """The message is not decoration: the correction loop reads it to repair the SQL.

    It reaches the loop through `repair_text` rather than `message`, because `message` is
    also what the caller is told and the source's words are made of the caller's schema.
    """
    r = prove(_Explains(fail='relation "ghost" does not exist'), "SELECT 1 FROM ghost")
    assert isinstance(r, Refusal) and r.code is RefusalCode.EXPLAIN_FAILED
    assert 'relation "ghost" does not exist' in r.repair_text


def test_a_validate_rejection_takes_the_same_path():
    r = prove(_Validates(fail="ORA-00942: table or view does not exist"), "SELECT 1 FROM ghost")
    assert isinstance(r, Refusal) and r.code is RefusalCode.EXPLAIN_FAILED
    assert "ORA-00942" in r.repair_text


def test_the_seam_is_optional_rather_than_required():
    """A required method would break every test double that stands in for an adapter, and the
    breakage would surface as a source error rather than as a missing method."""

    class _Bare:
        def execute(self, sql):
            return []

    assert prove(_Bare(), "SELECT 1") is None


def test_the_sources_words_are_kept_out_of_the_callers_half_of_a_refusal():
    """`message` is forwarded to the caller; the source's exception must not ride in it.

    `plan_query` ends an exhausted repair loop with `reason = last.message`, and that reason
    becomes `AgentAnswer.answer`. So anything `prove` interpolates into `message` is shipped
    to whoever asked -- and a source rejection names the relation and column it refused and
    can carry a DSN. The repair loop still needs those words, which is why they move to
    `source_detail` rather than being dropped.
    """
    leaky = ('permission denied for table hr_prod.payroll_salary; '
             'connection postgresql://svc_mnemiq@10.2.0.7:5432/hr_prod')
    verdict = prove(_Explains(fail=leaky), "SELECT n FROM claim")

    assert isinstance(verdict, Refusal)
    assert verdict.code == RefusalCode.EXPLAIN_FAILED
    for secret in ("payroll_salary", "hr_prod", "svc_mnemiq", "10.2.0.7", "postgresql://"):
        assert secret not in verdict.message, f"the caller-facing message disclosed {secret!r}"
    # The model is the consumer this half is FOR -- the repair loop is corrected by it. That is
    # not the same as the prompt being a safe place: see the residual at `Refusal.source_detail`.
    assert leaky in verdict.source_detail
    assert leaky in verdict.repair_text


def test_a_refusal_we_authored_still_reaches_the_caller_whole():
    """The split must not silence the refusals that are useful to read.

    Only `prove` interpolates a source exception. Every other refusal is text this engine
    wrote for a person, so `repair_text` and `message` stay the same string and the caller
    keeps the explanation.
    """
    ours = Refusal(code=RefusalCode.SELECT_STAR, message="SELECT * is not allowed.")

    assert ours.source_detail is None
    assert ours.repair_text == ours.message == "SELECT * is not allowed."
