"""A held table is matched by its parts, not by its parts joined with dots (M226).

A table's key joins catalog, schema and name with dots, so a table stored as `"aux.secret"` and
the reference `aux.secret`, table `secret` in schema `aux`, had one key. A grant on the first
approved the second, and the engine read the other object: on SQLite with a database attached as
`aux`, `SELECT v FROM aux.secret` returned the attached table's rows (measured). A key's parts are
known from the deployment: a single source keys a table by its name alone, whatever the name
holds, and a federated one by `catalog.table`, the catalog one the registry names. A reference is
held only where its own parts are that key's parts.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

DOTTED = {"aux.secret": {"v"}, "scratch": {"v"}}


def _sqlite():
    """A main database holding a table named `aux.secret` (row 1), and a second database attached
    as `aux` holding a table `secret` (row 999)."""
    con = sqlite3.connect(":memory:")
    con.execute('CREATE TABLE "aux.secret" (v INT)')
    con.execute('INSERT INTO "aux.secret" VALUES (1)')
    con.execute("CREATE TABLE scratch (v INT)")
    con.execute("ATTACH DATABASE ':memory:' AS aux")
    con.execute("CREATE TABLE aux.secret (v INT)")
    con.execute("INSERT INTO aux.secret VALUES (999)")
    return con


def test_the_engine_reads_the_two_spellings_as_two_tables():
    # The control: what each spelling returns, so a refusal below is of a real second object.
    con = _sqlite()
    assert con.execute('SELECT v FROM "aux.secret"').fetchall() == [(1,)]
    assert con.execute("SELECT v FROM aux.secret").fetchall() == [(999,)]


def test_the_held_table_written_as_one_name_is_approved_and_reads_its_own_rows():
    verdict = decide('SELECT v FROM "aux.secret"', DOTTED, dialect="sqlite", target="sqlite")
    assert isinstance(verdict, Approved), verdict
    assert _sqlite().execute(verdict.target_sql).fetchall() == [(1,)]


@pytest.mark.parametrize("sql", [
    "SELECT v FROM aux.secret",
    "SELECT s.v FROM aux.secret AS s",
    'SELECT v FROM "aux"."secret"',
    "SELECT v FROM scratch WHERE v IN (SELECT v FROM aux.secret)",
    "WITH t AS (SELECT v FROM aux.secret) SELECT v FROM t",
])
def test_a_reference_in_two_parts_is_not_the_table_named_with_a_dot(sql):
    verdict = decide(sql, DOTTED, dialect="sqlite", target="sqlite")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_TABLE, verdict
    # Never offered as the held table: it is another object, and the repair would say it is.
    assert "names the table listed" not in verdict.message, verdict.message


@pytest.mark.parametrize("sql", [
    "INSERT INTO aux.secret (v) VALUES (5)",
    "UPDATE aux.secret SET v = 0 WHERE v = 1",
    "DELETE FROM aux.secret WHERE v = 1",
    "INSERT INTO scratch (v) SELECT v FROM aux.secret",
])
def test_a_write_in_two_parts_is_not_the_table_named_with_a_dot(sql):
    grants = GrantSet(frozenset(DOTTED), writable=frozenset(DOTTED))
    verdict = decide_write(sql, DOTTED, grants, dialect="sqlite", target="sqlite",
                           policy=AccessPolicy(policy_schema=DOTTED), writes_enabled=True)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_TABLE, verdict


def test_a_write_to_the_held_table_written_as_one_name_is_approved():
    grants = GrantSet(frozenset(DOTTED), writable=frozenset(DOTTED))
    verdict = decide_write('INSERT INTO "aux.secret" (v) VALUES (5)', DOTTED, grants, dialect="sqlite",
                           target="sqlite", policy=AccessPolicy(policy_schema=DOTTED), writes_enabled=True)
    assert isinstance(verdict, ApprovedWrite), verdict
    con = _sqlite()
    con.execute(verdict.target_sql)
    assert con.execute("SELECT v FROM aux.secret").fetchall() == [(999,)]  # the attached table untouched


def test_oracle_matches_a_dotted_table_by_its_parts_too():
    visible = {"AUX.SECRET": {"V"}}
    assert isinstance(decide('SELECT V FROM "AUX.SECRET"', visible, dialect="oracle", target="oracle"), Approved)
    verdict = decide("SELECT V FROM AUX.SECRET", visible, dialect="oracle", target="oracle")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_TABLE, verdict


REGISTRY = {"pg": "public"}


def test_a_federated_table_is_its_catalog_and_its_name():
    visible = {"pg.claim": {"id"}, "pg.sales.2026": {"id"}}
    for sql in ("SELECT id FROM pg.claim", 'SELECT id FROM pg."sales.2026"'):
        verdict = decide(sql, visible, dialect="duckdb", target="duckdb", registry=REGISTRY)
        assert isinstance(verdict, Approved), (sql, verdict)


@pytest.mark.parametrize("sql", [
    'SELECT id FROM "pg.claim"',          # one name, read in the default catalog
    'SELECT id FROM pg.sales."2026"',     # catalog pg, schema sales, table 2026
    'SELECT id FROM "pg.sales"."2026"',
])
def test_a_federated_reference_whose_parts_are_not_the_key_s_is_refused(sql):
    visible = {"pg.claim": {"id"}, "pg.sales.2026": {"id"}}
    verdict = decide(sql, visible, dialect="duckdb", target="duckdb", registry=REGISTRY)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_TABLE, verdict


def test_a_key_under_a_catalog_the_registry_does_not_name_is_not_split():
    # A snapshot listing a catalog the deployment no longer attaches: nothing says how `ops.claim`
    # splits, so a reference in two parts is not decided against it.
    visible = {"pg.claim": {"id"}, "ops.claim": {"id"}}
    verdict = decide("SELECT id FROM ops.claim", visible, dialect="duckdb", target="duckdb", registry=REGISTRY)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_TABLE, verdict


def test_a_federated_write_is_decided_with_the_registry():
    visible = {"pg.claim": {"id"}}
    grants = GrantSet(frozenset(visible), writable=frozenset(visible))

    def _write(registry):
        return decide_write("DELETE FROM pg.claim WHERE id = 1", visible, grants, dialect="duckdb",
                            target="duckdb", policy=AccessPolicy(policy_schema=visible),
                            writes_enabled=True, registry=registry)
    assert isinstance(_write(REGISTRY), ApprovedWrite), _write(REGISTRY)
    refused = _write(None)  # read as a single source, `pg.claim` is two parts of a one-part key
    assert isinstance(refused, Refusal) and refused.code is RefusalCode.UNKNOWN_TABLE, refused


def test_check_access_without_a_registry_reads_a_single_source():
    # No registry is a single source, as `decide` reads it: a key is the table's name alone.
    ast = sqlglot.parse_one("SELECT id FROM pg.claim", read="duckdb")
    refused = check_access(ast, {"pg.claim": {"id"}})
    assert refused is not None and refused.code is RefusalCode.UNKNOWN_TABLE, refused
    assert check_access(ast, {"pg.claim": {"id"}}, registry=REGISTRY) is None


def test_the_runtime_s_write_passes_the_federation_s_registry(monkeypatch):
    # The join: a federated write decided as a single source would refuse every catalog-qualified
    # target, and one decided with no registry at all was the gap.
    import mnemiq.runtime as runtime

    seen = {}

    def _decide_write(sql, visible, grants, **kwargs):
        seen.update(kwargs)
        return Refusal(code=RefusalCode.NOT_A_WRITE, message="stop here")

    monkeypatch.setattr(runtime, "decide_write", _decide_write)

    class _Federated:
        dialect = "duckdb"
        registry = {"pg": "public"}

    class _Grants:
        def grants_for(self, identity):
            return GrantSet(frozenset({"pg.claim"}), writable=frozenset({"pg.claim"}))

    rt = runtime.Runtime.__new__(runtime.Runtime)
    rt.adapter, rt.snapshot, rt.settings, rt.authz = _Federated(), None, None, _Grants()
    rt.write("DELETE FROM pg.claim WHERE id = 1", identity=None)
    assert seen.get("registry") == {"pg": "public"}, seen
