"""A MAP literal's bare key is the column DuckDB reads, not the struct name sqlglot parses, at
every parse the decider makes: the plan's, a view body's, and the round trip that renders what
runs (Codex's forty-third review of #160, found on a row filter and traced to the shared seam).
`map_keys(MAP {secret: 1})[1]` returns the column's value on DuckDB; the grant check walked
Columns and saw a struct's name, and the plan's round trip to the target wrote `MAP {'secret':
1}`, a string, so the statement run was not the statement checked, and `MAP {id: 1}[id]`
answered for the string where the author asked for the column."""

from __future__ import annotations

import duckdb
import pytest
import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import ViewDefinition
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.reading import dotted_map_keys, unprintable
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode
from mnemiq.sql.views import body_of

VISIBLE = {"claim": {"id", "region", "tags", "x"}, "sink": {"i", "s"}}


class _OkAdapter:
    def execute(self, sql):
        return []


def _table():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, secret VARCHAR, tags VARCHAR[], x VARCHAR, attrs JSON); "
                "INSERT INTO claim VALUES (1, 'north', 's3cret', ['north'], 'col', '{\"a\": 1}'); "
                "CREATE TABLE sink(i INT, s VARCHAR); INSERT INTO sink VALUES (1, 'a')")
    return con


def _decide(sql):
    return decide(sql, VISIBLE, dialect="duckdb", target="duckdb")


def test_a_hidden_column_named_as_a_map_key_is_a_read():
    con = _table()
    sql = "SELECT map_keys(MAP {secret: 1})[1] FROM claim"
    assert con.execute(sql).fetchall() == [("s3cret",)]
    for spelling in (sql, 'SELECT map_keys(MAP {"secret": 1})[1] FROM claim', "SELECT (MAP {secret: 1})[1] FROM claim",
                     "SELECT MAP {secret: 1}[1] FROM claim"):
        verdict = _decide(spelling)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (spelling, verdict)
    # a struct literal's key is a name to DuckDB, no read
    assert con.execute("SELECT {secret: 1}.secret FROM claim").fetchall() == [(1,)]
    assert isinstance(_decide("SELECT {secret: 1}.secret FROM claim"), Approved)


def test_a_granted_map_key_runs_as_written():
    con = _table()
    for sql, written in (("SELECT MAP {id: 1}[id] FROM claim", "MAP {id: 1}"),
                         ("SELECT map_keys(MAP {region: 1})[1] FROM claim", "MAP {region: 1}")):
        rows = con.execute(sql).fetchall()
        verdict = _decide(sql)
        assert isinstance(verdict, Approved), verdict
        assert written in verdict.plan_sql and written in verdict.target_sql, verdict.target_sql
        assert con.execute(verdict.target_sql).fetchall() == rows, verdict.target_sql
    # the round trip under the row cap (a LIMIT sqlglot cannot parse after GROUPING SETS) reads the same
    sql = "SELECT MAP {id: 1}[id] AS k FROM claim GROUP BY GROUPING SETS ((id))"
    rows = con.execute(sql).fetchall()
    verdict = _decide(sql)
    assert isinstance(verdict, Approved) and "MAP {id: 1}" in verdict.target_sql, verdict
    assert con.execute(verdict.target_sql).fetchall() == rows


def test_a_write_renders_its_map_key_as_written_and_reads_it():
    grants = GrantSet(frozenset(VISIBLE), writable=frozenset({"sink"}))
    sql = "UPDATE sink SET s = 'b' WHERE MAP {i: 1}[i] = 1"
    verdict = decide_write(sql, VISIBLE, grants, adapter=_OkAdapter(), dialect="duckdb", target="duckdb", writes_enabled=True,
                           policy=AccessPolicy())
    assert isinstance(verdict, ApprovedWrite) and "MAP {i: 1}" in verdict.target_sql, verdict
    con = _table()
    con.execute(verdict.target_sql)
    assert con.execute("SELECT s FROM sink").fetchall() == [("b",)]
    verdict = decide_write("UPDATE sink SET s = 'b' WHERE MAP {secret: 1}[1] IS NULL", VISIBLE, grants, adapter=_OkAdapter(),
                           dialect="duckdb", target="duckdb", writes_enabled=True, policy=AccessPolicy())
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict


def test_a_view_bodys_map_key_is_a_column():
    body = body_of(ViewDefinition(object_id="leak", dialect="duckdb",
                                  definition="SELECT map_keys(MAP {secret: 1})[1] AS k FROM claim"))
    assert body is not None and {c.name for c in body.find_all(exp.Column)} == {"secret"}


def test_a_map_key_naming_its_lambdas_parameter_is_the_parameter():
    # The review gate on round 53: read as a Column, the key named no column of the table and a
    # valid query was refused. DuckDB binds the parameter before any column, a column `x` beside it
    # included (measured), so the key is a Var: bare in the render, no Column to the guards. A
    # parameter spelled in another case stays a Column and is refused, as it is elsewhere, since
    # DuckDB binds it to a column of the folded name in a SELECT list where one is in scope.
    con = _table()
    for sql in ("SELECT list_transform(tags, x -> MAP {x: 1}[x]) FROM claim",
                "SELECT list_transform([7], x -> MAP {x: 1}[7]) FROM claim"):
        rows = con.execute(sql).fetchall()
        verdict = _decide(sql)
        assert isinstance(verdict, Approved) and "MAP {x: 1}" in verdict.target_sql, (sql, verdict)
        assert con.execute(verdict.target_sql).fetchall() == rows, verdict.target_sql
    assert rows == [([1],)]  # the parameter 7, not the column 'col'
    sql = "SELECT list_transform(tags, x -> MAP {X: 1}[x]) FROM claim"
    verdict = _decide(sql)  # the column `x` of the folded name, granted: read as DuckDB reads it, and run as written
    assert isinstance(verdict, Approved) and "MAP {X: 1}" in verdict.target_sql, verdict
    assert con.execute(verdict.target_sql).fetchall() == con.execute(sql).fetchall() == [([None],)]
    verdict = decide(sql, {"claim": {"id", "region", "tags"}}, dialect="duckdb", target="duckdb")  # no such column granted
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    grants = GrantSet(frozenset(VISIBLE), writable=frozenset({"sink"}))
    verdict = decide_write("UPDATE sink SET s = 'b' WHERE list_transform([i], x -> MAP {x: 1}[x])[1] = 1", VISIBLE, grants,
                           adapter=_OkAdapter(), dialect="duckdb", target="duckdb", writes_enabled=True, policy=AccessPolicy())
    assert isinstance(verdict, ApprovedWrite) and "MAP {x: 1}" in verdict.target_sql, verdict


def test_a_quoted_parameter_key_keeps_its_quotes_in_the_render():
    # Codex's forty-sixth review of #160: the parameter's Var rendered its bare name, so `"x.secret"`
    # became `x.secret`, which the next parse reduced to `secret`, and `MAP {secret: 1}` ran against
    # the column the grant withholds where the text read the parameter; a doubled quote tokenized
    # nowhere and raised out of the decider. The Var carries the key's own Identifier: the name
    # compares bare, the render keeps the quotes, and the target runs to the text's rows.
    con = _table()
    for name in ('"x.secret"', '"a""b"', '"x y"', '"end"', '"X"'):
        sql = f"SELECT list_transform(tags, {name} -> map_keys(MAP {{{name}: 1}})[1]) FROM claim"
        rows = con.execute(sql).fetchall()
        assert rows == [(['north'],)], sql
        verdict = _decide(sql)
        assert isinstance(verdict, Approved) and f"MAP {{{name}: 1}}" in verdict.target_sql, (sql, verdict)
        assert con.execute(verdict.target_sql).fetchall() == rows, verdict.target_sql


def test_a_map_key_the_target_cannot_print_is_refused_not_raised():
    # Codex's forty-seventh review of #160: sqlglot's Postgres and Oracle generators name a struct's
    # field by a plain identifier, so the Column or the Var a MAP literal's key becomes raised a
    # ValueError out of both deciders for those targets. The plan is refused, repairably, and a
    # string key prints as before; the write path the same.
    for target in ("postgres", "oracle"):
        for sql in ('SELECT list_transform(tags, "x.secret" -> map_keys(MAP {"x.secret": 1})[1]) FROM claim',
                    "SELECT list_transform(tags, x -> MAP {x: 1}[x]) FROM claim",
                    "SELECT MAP {region: 1}[region] FROM claim"):
            verdict = decide(sql, VISIBLE, dialect="duckdb", target=target)
            assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNPRINTABLE_FOR_TARGET, (target, sql, verdict)
            assert verdict.repairable and target in verdict.message and "MAP literal" in verdict.message, verdict
        verdict = decide("SELECT MAP {'a': 1}['a'] FROM claim", VISIBLE, dialect="duckdb", target=target)
        assert isinstance(verdict, Approved), verdict
        grants = GrantSet(frozenset(VISIBLE), writable=frozenset({"sink"}))
        verdict = decide_write("UPDATE sink SET s = 'b' WHERE MAP {i: 1}[i] = 1", VISIBLE, grants, adapter=_OkAdapter(),
                               dialect="duckdb", target=target, writes_enabled=True, policy=AccessPolicy())
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNPRINTABLE_FOR_TARGET, (target, verdict)


def test_a_construct_the_target_cannot_render_is_refused_not_blanked():
    # M216 (Codex's forty-eighth review of #160, a side note): sqlglot's Postgres generator has no
    # `map_from_entries`, logged it as unsupported and wrote `SELECT  AS m FROM claim`, a statement
    # that is not the one decided, which the decider approved. Told to raise, the generator's
    # refusal is the unprintable one, without the MAP advice since the statement holds no literal;
    # a lambda Postgres renders is approved as before, and DuckDB runs the call as written.
    con = _table()
    sql = "SELECT map_from_entries([{'k': 'a', 'v': 1}]) AS m FROM claim"
    assert con.execute(sql).fetchall() == [({'a': 1},)]
    verdict = decide(sql, VISIBLE, dialect="duckdb", target="postgres")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNPRINTABLE_FOR_TARGET, verdict
    assert "MAP literal" not in verdict.message and "postgres" in verdict.message, verdict
    assert isinstance(decide(sql, VISIBLE, dialect="duckdb", target="duckdb"), Approved)
    verdict = decide("SELECT list_transform(tags, x -> x || 'a') FROM claim", VISIBLE, dialect="duckdb", target="postgres")
    assert isinstance(verdict, Approved) and "TRANSFORM(" in verdict.target_sql, verdict


def test_the_unprintable_advice_names_the_map_literal_only_where_the_statement_holds_one():
    # The review gate on rounds 61 and 62: the row-cap fallback's re-read and the write decider's
    # plain re-parse reach the same refusal with sqlglot's own errors, and advice about a MAP
    # literal the statement does not hold would spend the repair pass. The statement decides.
    with_map = sqlglot.parse_one("SELECT MAP {region: 1}[region] FROM claim", read="duckdb")
    without = sqlglot.parse_one("SELECT region FROM claim", read="duckdb")
    for exc in (ValueError("Name needs to be a string or an Identifier"), sqlglot.errors.ParseError("x"),
                sqlglot.errors.TokenError("x")):
        named = unprintable("postgres", exc, policys=False, tree=with_map)
        assert named.code is RefusalCode.UNPRINTABLE_FOR_TARGET and "MAP literal" in named.message and "postgres" in named.message
        bare = unprintable("postgres", exc, policys=False, tree=without)
        assert bare.code is RefusalCode.UNPRINTABLE_FOR_TARGET and "MAP" not in bare.message and "postgres" in bare.message
    policys = unprintable("oracle", ValueError("x"), policys=True, tree=with_map)
    assert policys.code is RefusalCode.POLICY_TARGET_UNSUPPORTED and not policys.repairable and "oracle" in policys.message


def test_an_arrow_outside_a_lambda_taker_is_no_lambda():
    # The review gate on round 54: sqlglot reads `a -> b` as a lambda inside any call, DuckDB
    # only inside the calls it hands a lambda to and as the JSON arrow elsewhere, so a MAP key
    # spelled as the arrow's left side was a Var no guard saw, and `concat(secret ->
    # map_keys(MAP {secret: 1})[1])` read the hidden column (DuckDB raises quoting its value).
    # Inside a real lambda the same spelling is the parameter, which shadows the column.
    con = _table()
    sql = "SELECT concat(secret -> map_keys(MAP {secret: 1})[1]) FROM claim"
    with pytest.raises(duckdb.Error, match="s3cret"):
        con.execute(sql).fetchall()
    verdict = _decide(sql)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    sql = "SELECT list_transform(tags, secret -> MAP {secret: 1}[secret]) FROM claim"
    assert con.execute(sql).fetchall() == [([1],)]
    verdict = _decide(sql)
    assert isinstance(verdict, Approved) and "MAP {secret: 1}" in verdict.target_sql, verdict


def test_an_arrow_outside_a_lambda_taker_is_the_json_extract_it_is():
    # M213 (the review gate on round 54): the arrow's left side is a column to DuckDB and a
    # lambda's parameter to sqlglot, so `concat(secret -> 1)` read the hidden column past the
    # grant check and DuckDB raised quoting its value; a row of columns the same. Re-read as the
    # JSON extract over Columns, the read is refused; over a granted JSON column the extract is
    # approved, rendered as written, and runs to the engine's rows.
    con = _table()
    with pytest.raises(duckdb.Error, match="s3cret"):
        con.execute("SELECT concat(secret -> 1) FROM claim").fetchall()
    for sql in ("SELECT concat(secret -> 1) FROM claim", "SELECT concat((id, secret) -> 1) FROM claim"):
        verdict = _decide(sql)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)
    sql = "SELECT concat(attrs -> '$.a', 'x') FROM claim"
    rows = con.execute(sql).fetchall()
    verdict = decide(sql, {**VISIBLE, "claim": VISIBLE["claim"] | {"attrs"}}, dialect="duckdb", target="duckdb")
    assert isinstance(verdict, Approved) and "attrs -> '$.a'" in verdict.target_sql, verdict
    assert con.execute(verdict.target_sql).fetchall() == rows == [("1x",)]


def test_an_arrow_in_a_lambda_takers_list_slot_is_the_json_extract_too():
    # The review gate on round 55: the taker check read the parent call and not the slot, so an
    # arrow in the list argument of a taker kept its parameter and the hidden column's read went
    # unseen (DuckDB reads the arrow there, then refuses the statement at bind for the types it
    # has). The lambda's own slot, the second argument, is a lambda still.
    for sql in ("SELECT list_transform(secret -> ['$'], x -> x) FROM claim",
                "SELECT len(list_filter(secret -> ['$'], x -> 1 = 1)) FROM claim",
                "SELECT list_reduce(secret -> ['$'], (a, b) -> a, 0) FROM claim"):
        verdict = _decide(sql)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)
    con = _table()
    sql = "SELECT list_transform(tags, x -> x) FROM claim"
    assert con.execute(sql).fetchall() == [(["north"],)]
    assert isinstance(_decide(sql), Approved)


def test_a_dotted_map_key_is_refused_before_the_parse_loses_it():
    # Codex's forty-fourth review of #160: sqlglot reads `MAP {x.region: 1}` as `MAP {region: 1}`
    # before any node exists, so the re-reading bound the key to the base column and the installed
    # statement answered another question (DuckDB reads the element's field). The text's tokens say
    # which keys the parse has lost, and the statement is refused with the spellings sqlglot keeps.
    # (The parameter is `e`: beside the fixture's column `x`, DuckDB binds `x.region` to the column.)
    con = _table()
    dotted = "SELECT list_transform([{region: 'south'}], e -> MAP {e.region: 1}[region]) FROM claim"
    kept = "SELECT list_transform([{region: 'south'}], e -> MAP {(e.region): 1}[region]) FROM claim"
    assert con.execute(dotted).fetchall() == con.execute(kept).fetchall() == [([None],)]
    verdict = _decide(dotted)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL, verdict
    assert "MAP {(e.region): ...}" in verdict.message and "MAP([e.region], [...])" in verdict.message, verdict.message
    # Only a chain of names is lost; a key sqlglot keeps whole is read as before (the gate's round).
    for sql, rows in (("SELECT MAP {claim.region::VARCHAR: 1}['north'] FROM claim", [(1,)]),
                      ("SELECT list_transform([[{f: 'a'}]], e -> MAP {e[1].f: 1}['a']) FROM claim", [([1],)]),
                      ("SELECT MAP {CASE WHEN claim.id > 0 THEN 'k' END: 1}['k'] FROM claim", [(1,)])):
        assert con.execute(sql).fetchall() == rows, sql
        verdict = _decide(sql)
        assert isinstance(verdict, Approved), (sql, verdict)
        assert con.execute(verdict.target_sql).fetchall() == rows, verdict.target_sql
    verdict = _decide(kept)
    assert isinstance(verdict, Approved) and "(e.region)" in verdict.target_sql, verdict
    assert con.execute(verdict.target_sql).fetchall() == [([None],)]
    grants = GrantSet(frozenset(VISIBLE), writable=frozenset({"sink"}))
    verdict = decide_write("UPDATE sink SET s = 'b' WHERE list_transform([{i: 1}], e -> MAP {e.i: 1}[i])[1] = 1", VISIBLE, grants,
                           adapter=_OkAdapter(), dialect="duckdb", target="duckdb", writes_enabled=True, policy=AccessPolicy())
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL, verdict
    assert body_of(ViewDefinition(object_id="v", dialect="duckdb",
                                  definition="SELECT list_transform([{region: 'south'}], e -> MAP {e.region: 1}[region]) AS k FROM claim")) is None


@pytest.mark.parametrize("sql, keys", [
    ("MAP {x.region: 1}[region]", ["x.region"]),
    ("MAP {\"x\".region: 1}", ['"x".region']),
    ("MAP {\u00e9.region: 1}", ["\u00e9.region"]),            # Codex's forty-fifth review: an ASCII-only class let it through
    ("MAP {\u03bb.a.b: 1}", ["\u03bb.a.b"]),
    ("MAP {e\u0301.region: 1}", ["e\u0301.region"]),            # the gate on round 58: a decomposed letter, a VAR to the tokenizer
    ("MAP {\u0663a.region: 1}", ["\u0663a.region"]),              # a name starting with a non-ASCII digit, the same
    ("MAP {x.key: 1}", ["x.key"]),                                # a keyword spelled as a word is a name
    ("MAP {a.b.c: MAP {d.e: 1}}", ["a.b.c", "d.e"]),
    ("MAP {'a.b': 1, x.y: 2}", ["x.y"]),
    ("MAP {(x.region): 1}", []),
    ("MAP {x[1].f: 1}", []),
    ("MAP {x.f(1): 1}", []),
    ("MAP {claim.region::VARCHAR: 1}", []),
    ("MAP {CASE WHEN x.a THEN 1 END: 1}", []),
    ("MAP {'a'.b: 1}", []),                                       # a string literal is no name (the gate on round 57)
    ("{x.region: 1}", []),                                        # a struct's key DuckDB refuses itself
])
def test_the_dotted_keys_the_parse_loses_are_read_off_the_text(sql, keys):
    assert dotted_map_keys(sql, "duckdb") == keys


def test_a_unicode_dotted_map_key_is_refused_too():
    # Codex's forty-fifth review of #160: `MAP {\u00e9.region: 1}` under `\u00e9 -> ...` passed the text check,
    # whose name class was ASCII, and was installed as `MAP {region: 1}`, the policy reversed.
    con = _table()
    sql = "SELECT list_transform([{region: 'south'}], \u00e9 -> MAP {\u00e9.region: 1}[region]) FROM claim"
    assert con.execute(sql).fetchall() == [([None],)]
    verdict = _decide(sql)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL and "\u00e9.region" in verdict.message, verdict
    sql = "SELECT list_transform([{region: 'south'}], e\u0301 -> MAP {e\u0301.region: 1}[region]) FROM claim"
    assert con.execute(sql).fetchall() == [([None],)]
    verdict = _decide(sql)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_CALL, verdict
    verdict = _decide("SELECT MAP {\"my col\".region: 1}['k'] FROM claim")
    assert isinstance(verdict, Refusal) and 'MAP {("my col".region): ...}' in verdict.message, verdict
