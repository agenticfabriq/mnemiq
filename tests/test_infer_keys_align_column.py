"""`align_column`, which decides whether a CREATE OR REPLACE runs over a live table.

Testable at all only because the module's two driver-pulling imports are deferred now. It had
none while its SQL was being changed, which is how the bare-`TRY_CAST` defect survived in this
file after being fixed in its sibling.

The invariant these pin is the one that was violated twice and stated in prose both times: the
loss PROBE and the REWRITE must cast identically. A probe certifying
`TRY_CAST(TO_VARCHAR(x) AS t)` while `TRY_CAST(x AS t)` executes has verified something
adjacent to what runs.
"""

import re

import pytest

from scripts.infer_keys_spider2 import align_column


class FakeCursor:
    """Records every statement and answers the two shapes `align_column` asks for."""

    def __init__(self, non_null=10, converted=10, fractional=0, columns=("ID", "NAME")):
        self.sql: list[str] = []
        self._reply = (non_null, converted, fractional)
        self._columns = columns
        self._last = ""

    def execute(self, sql, params=None):
        self.sql.append(sql)
        self._last = sql

    def fetchone(self):
        return self._reply

    def fetchall(self):
        return [(c,) for c in self._columns]

    # -- helpers the assertions read ------------------------------------------------
    @property
    def probe(self) -> str:
        return next(s for s in self.sql if s.lstrip().upper().startswith("SELECT COUNT("))

    @property
    def rewrite(self) -> str | None:
        return next((s for s in self.sql if "CREATE OR REPLACE TABLE" in s), None)


def _casts(sql: str) -> set[tuple[str, str]]:
    """Every `TRY_CAST(source AS target)` as a (source, target) pair.

    Hand-scanned rather than regexed because the target may itself contain parentheses --
    `NUMBER(38,0)` -- and `[^)]*` stops at the first one whatever its quantifier does, the
    negated class being what ends the match rather than greed. That is how the first version
    of this helper silently compared truncated targets.
    """
    out: set[tuple[str, str]] = set()
    for start in (m for m in range(len(sql)) if sql.startswith("TRY_CAST(", m)):
        depth, i = 0, start + len("TRY_CAST(")
        while i < len(sql):
            if sql[i] == "(":
                depth += 1
            elif sql[i] == ")":
                if depth == 0:
                    break
                depth -= 1
            i += 1
        source, _, target = sql[start + len("TRY_CAST("):i].partition(" AS ")
        out.add((source.strip(), target.strip()))
    return out


def _run(**kw) -> tuple[bool, FakeCursor]:
    cur = FakeCursor(**kw)
    ok = align_column(cur, "SPIDER2", "DB-IMDB", "M_CAST", "ID", "NUMBER(38,0)")
    return ok, cur


def test_a_lossless_cast_rewrites_the_table():
    ok, cur = _run(non_null=10, converted=10, fractional=0)
    assert ok is True
    assert cur.rewrite is not None


def test_a_lossy_cast_returns_False_and_writes_NOTHING():
    """The whole point of the probe. A value that does not convert becomes NULL in a table
    that would then be overwritten, so the rewrite must not run at all."""
    ok, cur = _run(non_null=10, converted=7)
    assert ok is False
    assert cur.rewrite is None, "it overwrote the table after deciding the cast was lossy"


def test_a_fractional_source_is_refused_for_a_NUMBER_target():
    """TRY_CAST rounds 3.5 to 4 rather than failing, so a fractional column converts
    `cleanly` while changing every value. An identifier has no fractional part."""
    ok, cur = _run(non_null=10, converted=10, fractional=3)
    assert ok is False
    assert cur.rewrite is None


def test_the_probe_and_the_rewrite_cast_identically():
    """The invariant. Both were bare `TRY_CAST(x AS t)` once and both are wrapped now; what
    must never happen again is ONE of them changing. Compares the actual cast expressions
    rather than trusting the comments that say so in two files."""
    _, cur = _run()
    probe, rewrite = _casts(cur.probe), _casts(cur.rewrite)
    assert rewrite, "no TRY_CAST in the rewrite at all"
    # SOURCE AND TARGET BOTH. An earlier version compared only the source, so changing the
    # rewrite's target to VARCHAR left every test in this file green -- the invariant test
    # blind to the half that decides what the column becomes.
    #
    # Subset, not equality: the rewrite casts the key column to the target, while the probe
    # casts it to the target AND to FLOAT for the fractional check.
    assert rewrite <= probe, (
        f"the rewrite casts something the probe never measured: {rewrite - probe}"
    )


@pytest.mark.parametrize("statement", ["probe", "rewrite"])
def test_every_cast_reads_a_string_source(statement):
    """Snowflake's TRY_CAST takes a STRING expression. A numeric source reaches here whenever
    the two sides differ and neither is TEXT -- `_CHILD_TYPES` admits FLOAT/REAL/DOUBLE, so a
    FLOAT child against a NUMBER parent picks the FLOAT side. The raise was swallowed by the
    caller, leaving a foreign key undeclared for a reason nothing printed."""
    _, cur = _run()
    sql = getattr(cur, statement)
    bare = re.findall(r'TRY_CAST\("', sql)
    assert not bare, f"{statement} casts a column directly instead of through TO_VARCHAR"


def test_the_untouched_columns_are_carried_through_unchanged():
    """A rebuild that dropped a column would be silent and total. `NAME` is not the key."""
    _, cur = _run(columns=("ID", "NAME"))
    assert '"NAME"' in cur.rewrite
    assert 'TRY_CAST(TO_VARCHAR("NAME")' not in cur.rewrite, "cast a column it was not asked to"
