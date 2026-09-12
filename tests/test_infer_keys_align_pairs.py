"""`align_pairs`, which rewrites live tables to make a join's two sides share a type.

The defect this pins was found by review and deferred twice as needing a live warehouse. That
stopped being true when the module's driver imports were deferred: a fake cursor drives it
end to end, and the failure is a COUNT of CREATE OR REPLACE statements, not a value in
Snowflake.

The shape: `align_column` probes and rewrites in one call, so deciding a pair with
`all(align_column(...) for s in sides)` short-circuits AFTER the first side is already
rewritten. The lossy second side aborts the pair, `types[s]` never records the first, and the
VARCHAR pass recomputes `sides` from the stale type and rewrites that same table again.
"""

from scripts.infer_keys_spider2 import align_pairs


class FakeCursor:
    """Answers the probe per-column and records every statement.

    `lossless` maps (table, column, target) -> bool. Anything unlisted converts cleanly.
    """

    def __init__(self, columns=("ID",), lossless=None, raise_rewrite_of=None,
                 raise_probe_of=None):
        self.sql: list[str] = []
        self._columns = columns
        self._lossless = lossless or {}
        # A table whose CREATE OR REPLACE raises -- the server-error path, distinct from a
        # cast the probe already refused.
        self._raise_rewrite_of = raise_rewrite_of
        # A (table, target) whose PROBE raises. Nothing has been written when it does, which
        # is why it must be handled differently from a failing rewrite.
        self._raise_probe_of = raise_probe_of
        self._pending: tuple | None = None

    def execute(self, sql, params=None):
        self.sql.append(sql)
        if "CREATE OR REPLACE TABLE" in sql and self._raise_rewrite_of:
            if f'"{self._raise_rewrite_of}"' in sql.split(" AS ")[0]:
                raise RuntimeError("002003 (42S02): SQL compilation error")
        if sql.lstrip().upper().startswith("SELECT COUNT("):
            if self._raise_probe_of:
                table, target = self._raise_probe_of
                if f'"{table}"' in sql and f"AS {target})" in sql:
                    raise RuntimeError("100038 (22018): numeric value not recognized")
            self._pending = self._probe_reply(sql)

    def _probe_reply(self, sql):
        # (non_null, converted, fractional). A lossy column converts 7 of its 10 values.
        for (table, column, target), ok in self._lossless.items():
            # `AS {target})` and not bare `target`: VARCHAR is a substring of the
            # `TO_VARCHAR(...)` that EVERY probe emits, so keying on the bare name would mark
            # the NUMBER(38,0) probe lossy too and quietly exercise a different path.
            if (f'"{table}"' in sql and f'"{column}"' in sql
                    and f"AS {target})" in sql and not ok):
                return (10, 7, 0)
        return (10, 10, 0)

    def fetchone(self):
        return self._pending

    def fetchall(self):
        return [(c,) for c in self._columns]

    # -- what the assertions read ---------------------------------------------------
    @property
    def rewrites(self) -> list[str]:
        return [s for s in self.sql if "CREATE OR REPLACE TABLE" in s]

    def rewrites_of(self, table: str) -> list[str]:
        return [s for s in self.rewrites if f'"{table}"' in s]


PAIR = [("CHILD", "ID", "PARENT", "ID")]
TEXT_VS_NUMBER = {("CHILD", "ID"): "TEXT", ("PARENT", "ID"): "NUMBER"}


def test_a_pair_that_converts_cleanly_is_aligned_once():
    cur = FakeCursor()
    types = dict(TEXT_VS_NUMBER)
    assert align_pairs(cur, "DB", "SCH", PAIR, types) == 1
    # Only the TEXT side needs moving; PARENT is already NUMBER.
    assert len(cur.rewrites_of("CHILD")) == 1
    assert cur.rewrites_of("PARENT") == []
    assert types[("CHILD", "ID")] == "NUMBER"


def test_both_sides_move_when_neither_holds_the_target():
    """The function's PRIMARY case: two sides actually REWRITTEN in one pass.

    Two other tests give `sides` length 2, but both abort at the probe, so before this one
    the rewrite loop had never run with more than a single side. Mutating it to
    `for s in sides[:1]:` leaves the pair still mismatched and its key undeclarable, and
    every other test stays green.

    TEXT against FLOAT: neither is NUMBER, so both are cast to NUMBER(38,0).
    """
    cur = FakeCursor()
    types = {("CHILD", "ID"): "TEXT", ("PARENT", "ID"): "FLOAT"}
    assert align_pairs(cur, "DB", "SCH", PAIR, types) == 2
    assert len(cur.rewrites_of("CHILD")) == 1
    assert len(cur.rewrites_of("PARENT")) == 1
    assert types[("CHILD", "ID")] == types[("PARENT", "ID")] == "NUMBER"


def test_the_rewrite_casts_to_the_target_the_probe_approved():
    """Nothing asserted WHICH type the rewrite casts to. Passing a constant `"VARCHAR"` while
    the probe still used `target` left the whole suite green -- the column would become
    VARCHAR while `types` recorded NUMBER, against a NUMBER parent, which is the silent type
    mismatch this script exists to remove."""
    cur = FakeCursor()
    types = dict(TEXT_VS_NUMBER)
    align_pairs(cur, "DB", "SCH", PAIR, types)
    rewrite = cur.rewrites_of("CHILD")[0]
    assert "AS NUMBER(38,0))" in rewrite, rewrite
    assert "AS VARCHAR)" not in rewrite, "cast to a type the probe never measured"


def test_a_side_that_cannot_convert_leaves_BOTH_sides_untouched():
    """The bug. Both sides need moving for the NUMBER target, the second is lossy, and the
    first used to be rewritten before that was discovered."""
    cur = FakeCursor(lossless={("PARENT", "ID", "NUMBER(38,0)"): False})
    types = {("CHILD", "ID"): "TEXT", ("PARENT", "ID"): "FLOAT"}
    align_pairs(cur, "DB", "SCH", PAIR, types)
    number_rewrites = [s for s in cur.rewrites if "NUMBER(38,0)" in s]
    assert number_rewrites == [], (
        "a table was rewritten to NUMBER before the pair was known to convert"
    )


def test_no_table_is_ever_rewritten_twice():
    """The consequence that corrupts data, and it needs a FLOAT child specifically.

    A side is re-included in the VARCHAR pass only when its recorded type is not already
    TEXT. So the double rewrite needs a side that (a) converts to NUMBER, (b) is aborted by
    its partner, and (c) is not TEXT: a FLOAT child against a TEXT parent. The first draft of
    this test had the types the other way round, and PASSED against the shipped bug -- CHILD
    was TEXT, so the VARCHAR pass excluded it and nothing was rewritten twice.

    Here CHILD goes FLOAT -> NUMBER(38,0) in the aborted pass, is not recorded, and is then
    re-rendered as VARCHAR holding the NUMBER spelling (1.0 -> 1 -> '1') while PARENT keeps
    its original text. The pair still mismatches, having been rewritten twice to get there.
    """
    cur = FakeCursor(lossless={("PARENT", "ID", "NUMBER(38,0)"): False})
    types = {("CHILD", "ID"): "FLOAT", ("PARENT", "ID"): "TEXT"}
    align_pairs(cur, "DB", "SCH", PAIR, types)
    for table in ("CHILD", "PARENT"):
        assert len(cur.rewrites_of(table)) <= 1, (
            f"{table} was rewritten {len(cur.rewrites_of(table))} times: "
            f"{cur.rewrites_of(table)}"
        )


def test_the_fallback_to_TEXT_still_works_when_NUMBER_is_refused():
    """Refusing the numeric target must not lose the pair -- VARCHAR is the whole point of
    the second iteration. Only PARENT moves, since CHILD is already TEXT."""
    cur = FakeCursor(lossless={("PARENT", "ID", "NUMBER(38,0)"): False})
    types = {("CHILD", "ID"): "TEXT", ("PARENT", "ID"): "FLOAT"}
    assert align_pairs(cur, "DB", "SCH", PAIR, types) == 1
    assert len(cur.rewrites_of("PARENT")) == 1
    assert "VARCHAR" in cur.rewrites_of("PARENT")[0]
    assert types[("PARENT", "ID")] == "TEXT"


def test_a_rewrite_that_RAISES_abandons_the_pair_instead_of_retrying_it():
    """The exception path, which the two-phase split did not close.

    `sides` is computed from the `child_type`/`parent_type` locals read before any write,
    never from the updated `types`. So continuing to the VARCHAR target after a rewrite
    raised would reselect the side this pass had ALREADY rewritten and re-render it -- the
    same double rewrite, surviving on a path the short-circuit fix did not touch.

    FLOAT against DOUBLE: they DIFFER, so the pair is not skipped as already-agreeing, and
    neither is NUMBER, so both are selected. Both probes pass, CHILD is rewritten, PARENT's
    rewrite raises. With `continue` the VARCHAR pass reselects CHILD -- neither local says
    TEXT -- and rewrites it again; with `break` the pair is abandoned and CHILD keeps its
    single rewrite. (An earlier draft used FLOAT on both sides and never reached the loop at
    all: equal types `continue` before any work.)
    """
    cur = FakeCursor(raise_rewrite_of="PARENT")
    types = {("CHILD", "ID"): "FLOAT", ("PARENT", "ID"): "DOUBLE"}
    aligned = align_pairs(cur, "DB", "SCH", PAIR, types)
    # COUNTED AS IT HAPPENS. Adding `len(sides)` after the loop meant a raise part-way
    # skipped the increment entirely, so a pair that half-wrote reported zero -- and `main`
    # prints its "(N columns retyped)" suffix only when the count is non-zero, so the one
    # table that HAD been rebuilt left no trace for the operator at all.
    assert aligned == 1, f"a half-written pair reported {aligned} columns rebuilt"
    assert len(cur.rewrites_of("CHILD")) == 1, (
        f"CHILD was rewritten {len(cur.rewrites_of('CHILD'))} times after the pair failed: "
        f"{cur.rewrites_of('CHILD')}"
    )


def test_a_PROBE_that_raises_still_falls_through_to_the_text_target():
    """A probe raise has written NOTHING, so the pair has lost nothing and the text fallback
    is still worth trying. Abandoning it there was a regression introduced by the fix for the
    rewrite-raise case: one `try` around both gave the read-only probe the half-written
    rewrite's treatment, and every pair whose NUMBER probe raised silently lost its VARCHAR
    alignment -- a foreign key not declared, for a reason nothing prints.
    """
    cur = FakeCursor(raise_probe_of=("PARENT", "NUMBER(38,0)"))
    types = {("CHILD", "ID"): "FLOAT", ("PARENT", "ID"): "DOUBLE"}
    assert align_pairs(cur, "DB", "SCH", PAIR, types) == 2, "the pair lost its text fallback"
    for table in ("CHILD", "PARENT"):
        assert len(cur.rewrites_of(table)) == 1
        assert "AS VARCHAR)" in cur.rewrites_of(table)[0]
    assert types[("CHILD", "ID")] == types[("PARENT", "ID")] == "TEXT"


def test_a_pair_whose_rewrite_raises_does_not_stop_the_pairs_after_it():
    """Abandoning the PAIR must not abandon the run -- that is the whole reason the `except`
    is inside the pair loop rather than around it."""
    cur = FakeCursor(raise_rewrite_of="PARENT")
    types = {("CHILD", "ID"): "FLOAT", ("PARENT", "ID"): "DOUBLE",
             ("LATER", "ID"): "TEXT", ("OK", "ID"): "NUMBER"}
    pairs = [("CHILD", "ID", "PARENT", "ID"), ("LATER", "ID", "OK", "ID")]
    align_pairs(cur, "DB", "SCH", pairs, types)
    assert len(cur.rewrites_of("LATER")) == 1, "a later pair was skipped"


def test_a_pair_that_already_agrees_is_not_touched():
    cur = FakeCursor()
    types = {("CHILD", "ID"): "NUMBER", ("PARENT", "ID"): "NUMBER"}
    assert align_pairs(cur, "DB", "SCH", PAIR, types) == 0
    assert cur.rewrites == []


def test_types_is_updated_so_a_later_pair_sees_the_new_type():
    """`types` is the running record of what each column now holds. A second pair sharing a
    column must not re-align it, which is exactly what the stale-entry bug caused."""
    cur = FakeCursor()
    types = {("CHILD", "ID"): "TEXT", ("PARENT", "ID"): "NUMBER", ("OTHER", "ID"): "NUMBER"}
    pairs = [("CHILD", "ID", "PARENT", "ID"), ("CHILD", "ID", "OTHER", "ID")]
    align_pairs(cur, "DB", "SCH", pairs, types)
    assert len(cur.rewrites_of("CHILD")) == 1, "the second pair re-aligned an aligned column"
