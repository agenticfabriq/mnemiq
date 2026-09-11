from __future__ import annotations

import threading

import duckdb
import pyarrow as pa

# The declared-FK query for Postgres sources: run through postgres_query so the
# information_schema joins execute with real Postgres semantics, not DuckDB's proxy.
_PG_FK_QUERY = (
    "SELECT kcu.table_name, kcu.column_name, ccu.table_name, ccu.column_name, "
    "  tc.constraint_name "
    "FROM information_schema.table_constraints tc "
    "JOIN information_schema.key_column_usage kcu "
    "  ON tc.constraint_name = kcu.constraint_name "
    "  AND tc.table_schema = kcu.table_schema "
    "JOIN information_schema.constraint_column_usage ccu "
    "  ON tc.constraint_name = ccu.constraint_name "
    "  AND tc.table_schema = ccu.table_schema "
    "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public' "
    "ORDER BY kcu.table_name, kcu.ordinal_position"
)


# Views for a Postgres source, for the same reason as the FK query: DuckDB's proxy reports
# every attached-Postgres view as a BASE TABLE, so asking it finds nothing at all.
_PG_VIEW_QUERY = (
    "SELECT c.relname, pg_get_viewdef(c.oid, true) "
    "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
    "WHERE c.relkind = 'v' AND n.nspname = 'public' ORDER BY c.relname"
)


def duckdb_user_functions(con) -> list[str]:
    """Function names a DuckDB connection defines itself, anywhere in any attached catalog.

    `internal` is the discriminator and the whole reason this is answerable: DuckDB marks its own
    945 builtins `internal = true`, and anything a deployment adds comes back false.

    Module-level because `FederatedAdapter` is a DuckDB connection too -- ATTACH per source into
    one connection -- and a second copy of these three queries is a second thing to keep true.
    """
    return [r[0] for r in con.execute(
        "SELECT DISTINCT lower(function_name) FROM duckdb_functions() WHERE NOT internal"
    ).fetchall()]


def duckdb_builtin_functions(con) -> list[str]:
    """The other half of the same catalogue, split on the same flag."""
    return [r[0] for r in con.execute(
        "SELECT DISTINCT lower(function_name) FROM duckdb_functions() WHERE internal"
    ).fetchall()]


def duckdb_reachable_user_functions(con) -> list[str]:
    """Those an unqualified call can reach: the ones whose schema is on the search path."""
    path = con.execute("SELECT current_setting('search_path')").fetchall()[0][0]
    entries = {e.strip().strip('"').lower() for e in str(path).split(",") if e.strip()}
    return [fn for db, schema, fn in con.execute(
        "SELECT DISTINCT lower(database_name), lower(schema_name), lower(function_name) "
        "FROM duckdb_functions() WHERE NOT internal").fetchall()
        if f"{db}.{schema}" in entries or schema in entries]


class DuckDBAdapter:
    """DuckDB as the universal executor: ATTACH a source and read it via DuckDB's scanner.

    The engine's plan is written in duckdb and executed here, so DuckDB-native functions
    (YEAR, EXTRACT, LISTAGG) always work -- regardless of what the underlying source is.
    Build one via a factory: DuckDBAdapter.postgres(dsn) or DuckDBAdapter.sqlite(path).
    """

    dialect = "duckdb"

    def __init__(
        self,
        *,
        attach_target: str,
        attach_type: str,
        extension: str,
        catalog: str,
        table_schema: str,
        fk_via_postgres: bool,
        read_only: bool = True,
    ) -> None:
        self._catalog = catalog
        self._table_schema = table_schema
        self._fk_via_postgres = fk_via_postgres
        self._attach_type = attach_type.upper()
        self._con = duckdb.connect()
        # A DuckDB file needs no extension: the engine already speaks its own format. INSTALLing a
        # nonexistent "duckdb" extension would fail, so the empty string means "nothing to load".
        if extension:
            self._con.execute(f"INSTALL {extension}; LOAD {extension};")
        # READ_ONLY unless a write is explicitly enabled -- the backstop under the write path.
        clause = f"(TYPE {attach_type}, READ_ONLY)" if read_only else f"(TYPE {attach_type})"
        self._con.execute(f"ATTACH '{attach_target}' AS {catalog} {clause}")
        self._con.execute(f"USE {catalog}.{table_schema}")

    @classmethod
    def postgres(cls, dsn: str, schema: str = "src", read_only: bool = True) -> "DuckDBAdapter":
        return cls(attach_target=dsn, attach_type="POSTGRES", extension="postgres",
                   catalog=schema, table_schema="public", fk_via_postgres=True, read_only=read_only)

    @classmethod
    def duckdb(cls, path: str, schema: str = "d", read_only: bool = True) -> "DuckDBAdapter":
        """A DuckDB file as the source.

        The class calls itself the universal executor and could attach Postgres and SQLite and not
        DuckDB, so a warehouse that IS DuckDB had no adapter at all. `main` is DuckDB's default
        schema, and no extension is needed to read its own format.
        """
        return cls(attach_target=path, attach_type="DUCKDB", extension="",
                   catalog=schema, table_schema="main", fk_via_postgres=False,
                   read_only=read_only)

    @classmethod
    def sqlite(cls, path: str, schema: str = "s", read_only: bool = True) -> "DuckDBAdapter":
        return cls(attach_target=path, attach_type="SQLITE", extension="sqlite",
                   catalog=schema, table_schema="main", fk_via_postgres=False, read_only=read_only)

    def introspect(self) -> list[str]:
        rows = self._con.execute(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_catalog = '{self._catalog}' AND table_schema = '{self._table_schema}'"
        ).fetchall()
        return [r[0] for r in rows]

    def list_columns(self) -> list[tuple[str, str, str]]:
        rows = self._con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            f"WHERE table_catalog = '{self._catalog}' AND table_schema = '{self._table_schema}' "
            "ORDER BY table_name, ordinal_position"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        """Declared FKs: (from_table, from_col, to_table, to_col, constraint_id).

        Postgres exposes them via information_schema; DuckDB's SQLite scanner does not, so a
        SQLite source returns [] (enrichment falls back to data-driven inference). Any
        failure -> [] (fall back to inference), never an exception.
        """
        if not self._fk_via_postgres:
            return []
        try:
            rows = self._con.execute(
                f"SELECT * FROM postgres_query('{self._catalog}', $q${_PG_FK_QUERY}$q$)"
            ).fetchall()
        except Exception:
            return []
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    @property
    def functions_cover_view_bodies(self) -> bool:
        """Whether `user_functions()` also answers for what a VIEW BODY here may call.

        For a DuckDB file it does: bodies are DuckDB views and run against this catalogue. For a
        Postgres attachment it does not -- measured, a Postgres view body calling a Postgres UDF
        runs server-side and returns its value, where DuckDB's binder never looked.

        Keyed on the ATTACH TYPE, and only `DUCKDB` grants it. It used to read
        `not self._fk_via_postgres`, which is a flag named for foreign-key discovery: SQLite got
        the licence (harmless only because SQLite stores no user functions, an argument nothing
        wrote down), and a future `mysql()` classmethod would naturally pass
        `fk_via_postgres=False` and silently certify server-side bodies calling MySQL stored
        functions. An allowlist of one, so a new attachment kind gets the conservative answer
        without anyone having to notice.

        A property rather than a field, so a test can read the real derivation off a bare
        instance. As a field the only way to check it without a live Postgres was to restate the
        derivation in the test, which passes whatever `__init__` does.
        """
        return self._attach_type == "DUCKDB"

    def user_functions(self) -> list[str]:
        """Function names this database defines itself, from `duckdb_functions()`.

        `internal` is the discriminator, and it is the whole reason this is answerable here:
        DuckDB marks its own 945 builtins `internal = true`, and anything a deployment adds --
        a macro, a table macro -- comes back false. Measured on a fresh connection: `count` and
        `median` are internal, a `CREATE MACRO` is not.

        That settles the question parsing cannot. sqlglot types `median` as a builtin before the
        source binds it, so a UDF wearing that name is invisible to the decider (M43's residual)
        and forces lineage to decline certification on every call (issue #5).

        Names only, unqualified, because both callers ask "is this name bound to something this
        source defines" and a schema-qualified call resolves through the search path anyway.
        Duplicates collapse: DuckDB lists one row per overload.

        DuckDB's catalogue is the right one even for a Postgres attachment, which is not
        obvious and was measured rather than assumed. Queries execute on `self._con`, and
        DuckDB's binder resolves calls against its own catalogue: against a Postgres source
        holding `CREATE FUNCTION median(int) RETURNS int AS $$ SELECT 99 $$`, `SELECT median(1)`
        returned 1.0 from DuckDB's aggregate, and both `public.median(1)` and
        `src.public.median(1)` failed to bind. A Postgres UDF is simply not reachable from a
        generated query, so reporting DuckDB's non-internal names answers the question both
        callers ask. An earlier version raised here on the theory that this catalogue was the
        wrong one; it was throwing away a correct answer.

        What that does NOT cover is a Postgres VIEW BODY. Reading `src.public.v_med`, whose body
        calls that same function, returned 99 -- the function ran server-side, inside Postgres,
        where DuckDB's binder never looked. Lineage walking view bodies is asking a different
        question of a different catalogue, and wants `pg_proc`. Naming the gap here so the two
        do not get conflated.

        A failure RAISES, like `view_definitions`. Returning `[]` would tell the caller this
        source defines nothing, which is a different claim from being unable to look.
        """
        return duckdb_user_functions(self._con)

    def reachable_user_functions(self) -> list[str]:
        """The subset of `user_functions()` a query can call WITHOUT qualifying it.

        `duckdb_functions()` carries `database_name` and `schema_name` and the first version of
        `user_functions` threw both away, which made every name look reachable. Measured on an
        attached file holding `CREATE MACRO other.median(x)`: `SELECT median(id) FROM claim`
        returned **1.5**, DuckDB's builtin, because `other` is not on the search path -- and the
        engine refused the whole source, `SELECT id FROM claim` included, on the theory that
        `median` was shadowed. One unrelated macro in one unused schema made a legitimate source
        answer nothing.

        Only the coarse rule wants this. `names` stays the full list, because a query MAY
        qualify -- `other.median(1)` returns 999 -- and the bare name read off the rendered text
        is what catches that.

        The search path, not `current_schema()`: DuckDB resolves an unqualified function against
        every entry, and reading only the current one would call an entry further down the path
        unreachable. `database.schema` and a bare `schema` are both accepted because the setting
        holds either.
        """
        return duckdb_reachable_user_functions(self._con)

    def builtin_functions(self) -> list[str]:
        """The other half of `duckdb_functions()`: names this engine considers its own.

        Only ever used to intersect with `user_functions()`, and that intersection is the set of
        names a call can reach WITHOUT the query spelling them. Measured on an attached
        read-only DuckDB file holding `CREATE MACRO count_star() AS (SELECT ssn FROM secret)`:
        `SELECT count(*) FROM claim` returned the SSN. `count_star` is DuckDB's binder name for
        `COUNT(*)` and appears in no spelling of that query, in any dialect -- and it could only
        be reached because `count_star` is a builtin. A macro named `safe_helper` has no such
        route, which is why the intersection and not the mere presence of macros is the test.

        The same catalogue query as `user_functions`, so the two partitions are of one table and
        cannot disagree about what `internal` means. Two round trips rather than one because
        `user_functions` shipped first and its callers do not all want this; the window between
        them is the same staleness a single query already has against a source someone is
        editing.

        A failure RAISES, like its sibling, and `inventory_from` turns that into "builtins not
        known". Know what that costs before omitting this method: a source with any user
        function at all then has every read AND every write against it refused. It is the
        conservative reading and it is not a soft one.
        """
        return duckdb_builtin_functions(self._con)

    def view_definitions(self) -> list[tuple[str, str, str]]:
        """(view, body, dialect) for every view in the source. A failure RAISES.

        It used to return `[]` on any failure, and that single choice defeated the whole
        availability signal above it: `enrichment/pipeline.py` wraps this call in a `try/except`
        precisely so a source that will not answer records `discover:views` as `failed`, and it
        could never fire because the exception was eaten here. The job said `done`, the snapshot
        carried an empty `views`, and `inventory_for` reported a COMPLETE inventory of nothing --
        so a granted view over a filtered table was absent from it, its base filter was narrowed
        away, and the read was approved unfiltered. `SQLiteAdapter.view_definitions` never
        swallowed, which is why the test pinning that contract stayed green.

        An empty list still means "this source reports no views" and is answered honestly by a
        query that succeeds with no rows. What it must never mean again is "nobody could ask".

        **DuckDB's `information_schema` cannot answer this for an attached Postgres**: it
        reports every view as `BASE TABLE`, so the obvious discovery path finds no views at all
        and the governance that depends on knowing one silently does nothing. Measured on
        Pagila -- 7 views, 0 reported. So a Postgres source is asked through `postgres_query`,
        exactly as declared FKs already are, and the body comes back as Postgres SQL.

        A native DuckDB file answers correctly from `duckdb_views()`. A SQLite source attached
        through DuckDB's scanner exposes neither, and returns [] -- the same posture as
        `foreign_keys`, and the reason `SQLiteAdapter` answers for itself.
        """
        if self._fk_via_postgres:  # the source is Postgres, reached through the attachment
            try:
                rows = self._con.execute(
                    f"SELECT * FROM postgres_query('{self._catalog}', $q${_PG_VIEW_QUERY}$q$)"
                ).fetchall()
            except Exception as exc:  # the caller records `discover:views` failed
                raise RuntimeError(f"could not read view definitions from Postgres: {exc}") from exc
            return [(r[0], r[1], "postgres") for r in rows]
        try:
            rows = self._con.execute(
                "SELECT view_name, sql FROM duckdb_views() WHERE NOT internal "
                f"AND schema_name = '{self._table_schema}' ORDER BY view_name"
            ).fetchall()
        except Exception as exc:  # the caller records `discover:views` failed
            raise RuntimeError(f"could not read view definitions from DuckDB: {exc}") from exc
        return [(r[0], r[1], "duckdb") for r in rows]

    def execute(self, sql: str) -> list[tuple]:
        return self._con.execute(sql).fetchall()

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """Run a query and return Arrow, cancelling it if it overruns. A LIMIT bounds rows;
        only an interrupt bounds a query that never produces a first row."""
        timer = None
        if timeout_s is not None:
            timer = threading.Timer(timeout_s, self._con.interrupt)
            timer.start()
        try:
            return self._con.execute(sql).to_arrow_table()
        finally:
            if timer is not None:
                timer.cancel()


class DuckDBPostgresAdapter(DuckDBAdapter):
    """Compat: Query a Postgres source through DuckDB's postgres extension (ATTACH).

    Preserved as a thin subclass so existing imports and call sites are unchanged.
    """

    def __init__(self, dsn: str, schema: str = "src", read_only: bool = True) -> None:
        super().__init__(attach_target=dsn, attach_type="POSTGRES", extension="postgres",
                         catalog=schema, table_schema="public", fk_via_postgres=True,
                         read_only=read_only)
