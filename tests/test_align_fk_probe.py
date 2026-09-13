"""`probe_component`, the loss check that decides whether a component may be rewritten.

The defect it was extracted to fix: the probe ran PER TABLE and skipped that one table, which
split the very component the grouping exists to hold together. With `P.K` TEXT, `C1.K` TEXT
and `C2.K` NUMBER the plan moves both `P` and `C1` to NUMBER, so a `P` that fails the probe --
the population this script exists for, text columns holding the values that violate the
numeric type -- left `C1` rewritten to NUMBER against a still-TEXT parent. `C1` and `P` AGREED
before the run, and the operator saw two unrelated lines: one aligned, one skipped.

`infer_keys_spider2.align_pairs` states the discipline for the pair case -- decide both sides
before writing either. A component needs it at its own scale.
"""

from scripts.align_fk_types_snowflake import probe_component

NUM = "NUMBER(38,0)"
COMPONENT = {"P": {"K": NUM}, "C1": {"K": NUM}}


class FakeCursor:
    """Answers the loss probe per table. `lossy_tables` convert 7 of their 10 values."""

    def __init__(self, lossy_tables=(), raise_on=()):
        self.sql: list[str] = []
        self._lossy = set(lossy_tables)
        self._raise_on = set(raise_on)
        self._reply = None

    def execute(self, sql, params=None):
        self.sql.append(sql)
        table = next((t for t in ("P", "C1", "C2") if f'."{t}"' in sql), None)
        if table in self._raise_on:
            raise RuntimeError("100038 (22018): numeric value not recognized")
        self._reply = (10, 7, 0) if table in self._lossy else (10, 10, 0)

    def fetchone(self):
        return self._reply


def test_a_component_that_converts_cleanly_reports_nothing():
    assert probe_component(FakeCursor(), "DB", "SCH", COMPONENT) == []


def test_one_lossy_member_condemns_the_WHOLE_component():
    """The blocker. Only `P` is lossy, and the answer must be about the component -- returning
    nothing for `C1` is what let `C1` be rewritten around a skipped `P`."""
    lossy = probe_component(FakeCursor(lossy_tables={"P"}), "DB", "SCH", COMPONENT)
    assert lossy, "a lossy member reported nothing, so the component would be rewritten"
    assert any("P.K" in line for line in lossy), lossy


def test_the_reason_names_the_table_as_well_as_the_column():
    """A component spans tables, so `K (3 would become NULL)` does not say which `K`."""
    lossy = probe_component(FakeCursor(lossy_tables={"C1"}), "DB", "SCH", COMPONENT)
    assert lossy == ["C1.K (3 would become NULL)"], lossy


def test_a_probe_that_RAISES_condemns_the_component_rather_than_the_run():
    """A safety check must not be the thing that stops the run. An unreadable probe means the
    component cannot be SHOWN to convert -- a reason to leave it alone and say so, not to
    abandon the components after it."""
    lossy = probe_component(FakeCursor(raise_on={"P"}), "DB", "SCH", COMPONENT)
    assert any("P.K (probe failed:" in line for line in lossy), lossy


def test_a_probe_exception_with_an_EMPTY_message_does_not_end_the_run():
    """The probe handler exists so an unreadable column does not stop the run, and it built
    its message with `str(exc).splitlines()[-1]` -- which raises IndexError on an empty
    message, since `"".splitlines()` is `[]`. The IndexError then escaped the handler, the
    component loop and the schema loop, ending exactly what the handler was written to
    protect.

    A commit message claimed this idiom had already been routed through `_reason` everywhere.
    It had not: `apply_component` was converted and this one was left, so the claim was
    checkable and wrong.
    """

    class EmptyMessage(FakeCursor):
        def execute(self, sql, params=None):
            self.sql.append(sql)
            raise RuntimeError("")

    lossy = probe_component(EmptyMessage(), "DB", "SCH", COMPONENT)
    assert len(lossy) == 2, lossy
    assert all("probe failed:" in line for line in lossy), lossy


def test_a_fractional_source_is_refused_for_a_NUMBER_target():
    """TRY_CAST rounds 3.5 to 4 rather than failing, so a fractional column converts
    `cleanly` while changing every value. An identifier has no fractional part."""

    class Fractional(FakeCursor):
        def execute(self, sql, params=None):
            self.sql.append(sql)
            self._reply = (10, 10, 4)

    lossy = probe_component(Fractional(), "DB", "SCH", {"P": {"K": NUM}})
    assert lossy == ["P.K (4 fractional, TRY_CAST would round)"], lossy


def test_every_probe_reads_a_string_source():
    """Snowflake's TRY_CAST takes a STRING expression, and the probe must cast exactly what
    the rewrite casts -- a probe certifying `TRY_CAST(TO_VARCHAR(x) AS t)` while
    `TRY_CAST(x AS t)` executes has verified something adjacent to what runs."""
    cur = FakeCursor()
    probe_component(cur, "DB", "SCH", COMPONENT)
    assert cur.sql, "no probe was issued at all"
    for sql in cur.sql:
        assert 'TRY_CAST("' not in sql, f"cast a column directly: {sql}"


def test_every_member_is_probed_not_just_the_first():
    """A probe that stopped at the first clean table would clear a component whose second
    member cannot convert."""
    cur = FakeCursor()
    probe_component(cur, "DB", "SCH", COMPONENT)
    probed = {t for t in ("P", "C1") if any(f'."{t}"' in s for s in cur.sql)}
    assert probed == {"P", "C1"}, probed


# --------------------------------------------------------------------------------------------
# `build_rewrites` -- every READ for a component, before any of it is WRITTEN.
# --------------------------------------------------------------------------------------------


class ReadCursor:
    """Answers the INFORMATION_SCHEMA column read. `raise_on` tables fail it."""

    def __init__(self, columns=("K", "OTHER"), raise_on=()):
        self.sql: list[str] = []
        self._columns = columns
        self._raise_on = set(raise_on)

    def execute(self, sql, params=None):
        self.sql.append(sql)
        table = next((t for t in ("P", "C1") if f"table_name = '{t}'" in sql), None)
        if table in self._raise_on:
            raise RuntimeError("002003 (02000): Object does not exist")

    def fetchall(self):
        return [(c,) for c in self._columns]


def test_a_failed_read_propagates_for_the_caller_to_turn_into_a_skip():
    """`build_rewrites` only reads, so it cannot swallow a read failure into a partial plan.

    An earlier version of this test also asserted that nothing was written -- tautological,
    since this function issues one SELECT and never a CREATE OR REPLACE, so no version of it
    could fail that. The property the hoist actually establishes lives in `apply_component`,
    and is tested there.
    """
    from scripts.align_fk_types_snowflake import build_rewrites

    try:
        build_rewrites(ReadCursor(raise_on={"C1"}), "DB", "SCH", COMPONENT)
    except RuntimeError:
        return
    raise AssertionError("the read failure was swallowed")


def test_every_table_of_the_component_gets_a_statement():
    from scripts.align_fk_types_snowflake import build_rewrites

    rewrites = build_rewrites(ReadCursor(), "DB", "SCH", COMPONENT)
    assert {t for t, _, _ in rewrites} == {"P", "C1"}
    assert all("CREATE OR REPLACE TABLE" in sql for _, sql, _ in rewrites)


def test_the_projection_casts_the_key_and_carries_the_rest_verbatim():
    """A rebuild that dropped a column would be silent and total, and a cast applied to the
    wrong one changes data nobody asked about."""
    from scripts.align_fk_types_snowflake import build_rewrites

    sql = dict((t, s) for t, s, _ in build_rewrites(ReadCursor(), "DB", "SCH", COMPONENT))["P"]
    assert f'TRY_CAST(TO_VARCHAR("K") AS {NUM}) AS "K"' in sql, sql
    assert '"OTHER"' in sql and 'TO_VARCHAR("OTHER")' not in sql, sql


# --------------------------------------------------------------------------------------------
# `apply_component` -- where the read hoist and the split report actually live.
# --------------------------------------------------------------------------------------------


class ApplyCursor:
    """Reads the column list, then executes rewrites. Either can be made to fail per table."""

    def __init__(self, raise_read=(), raise_write=(), columns=("K",)):
        self.sql: list[str] = []
        self._raise_read = set(raise_read)
        self._raise_write = set(raise_write)
        self._columns = columns

    def execute(self, sql, params=None):
        self.sql.append(sql)
        if "INFORMATION_SCHEMA" in sql:
            table = next((t for t in ("A", "B", "C") if f"table_name = '{t}'" in sql), None)
            if table in self._raise_read:
                raise RuntimeError("002003 (02000): Object does not exist")
            return
        table = next((t for t in ("A", "B", "C") if f'."{t}"' in sql), None)
        if table in self._raise_write:
            raise RuntimeError("000603 (XX000): execution error")

    def fetchall(self):
        return [(c,) for c in self._columns]

    @property
    def writes(self):
        return [s for s in self.sql if "CREATE OR REPLACE" in s]


THREE = {"A": {"K": NUM}, "B": {"K": NUM}, "C": {"K": NUM}}


def _apply(**kw):
    from scripts.align_fk_types_snowflake import apply_component

    cur = ApplyCursor(**kw)
    return apply_component(cur, "DB", "SCH", THREE), cur


def test_a_clean_component_writes_every_table():
    out, cur = _apply()
    assert [t for t, _ in out["written"]] == ["A", "B", "C"]
    assert out["failed"] is None and out["unreadable"] is None
    assert len(cur.writes) == 3


def test_A_READ_FAILURE_ON_A_LATER_TABLE_WRITES_NOTHING():
    """The one that used to end the run. The read sat inside the write loop and outside any
    `try`, so failing on `B` came AFTER `A` was rewritten -- and then escaped every loop.

    Hoisted: nothing is written at all, and the caller turns it into an UNREADABLE line.
    """
    out, cur = _apply(raise_read={"B"})
    assert cur.writes == [], f"it wrote before finishing the reads: {cur.writes}"
    assert out["unreadable"] and out["written"] == []


def test_a_write_failure_names_the_tables_left_behind_AND_the_ones_never_attempted():
    """`B` fails, so `A` is rewritten and BOTH `B` and `C` disagree with it -- `C` was never
    attempted. Reporting only the failure reads as one unlucky rewrite when the component is
    inconsistent three ways."""
    out, _ = _apply(raise_write={"B"})
    assert [t for t, _ in out["written"]] == ["A"]
    assert out["failed"][0] == "B"
    assert out["unattempted"] == ["C"], out["unattempted"]


def test_an_exception_with_an_EMPTY_message_does_not_end_the_run():
    """`str(exc).splitlines()[-1]` raises IndexError on an empty message, inside the handler
    whose whole purpose is to keep a failure from ending the run."""

    class Empty(ApplyCursor):
        def execute(self, sql, params=None):
            self.sql.append(sql)
            if "INFORMATION_SCHEMA" not in sql:
                raise RuntimeError("")

    from scripts.align_fk_types_snowflake import apply_component

    out = apply_component(Empty(), "DB", "SCH", THREE)
    assert out["failed"][0] == "A" and out["failed"][1], out["failed"]
