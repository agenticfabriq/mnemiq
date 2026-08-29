"""One answer to "which adapter does this source need", for every door that asks.

Three places built an adapter before this module existed -- `build_runtime`, the `enrich` command
and the `refresh`/diff command -- and all three answered the question the same way: construct a
`DuckDBPostgresAdapter` on `settings.pg_dsn`, unconditionally. That is fine while every source is
Postgres and wrong the moment one is not, in two separate ways that were both measured:

  1. **A one-entry manifest was silently discarded.** `Settings.source_specs` honours
     `MNEMIQ_SOURCES_PATH`, and `build_runtime` federates when it yields two or more specs -- but
     the single-source branch never reads `specs[0]`. A manifest naming one SQLite file resolved
     to a spec with `kind="sqlite"`, and the engine then connected to `pg_dsn` and never mentioned
     it. Two sources were honoured; one was not.
  2. **Enrichment could not follow.** Even a correctly dispatched runtime has nothing to plan
     against until `enrich` builds a snapshot, and `enrich` hardcoded the same Postgres adapter.
     Wiring only the runtime would have produced an engine that connects to Oracle and knows
     nothing about it -- a seam wired at one end, which is the failure this codebase keeps
     finding (see **M26**, where an emitter shipped with no caller and a green suite).

So the resolver is the unit, not the Oracle branch. Adding a `kind` in one door and not the others
is how the two bugs above were written in the first place.

**Oracle credentials come from settings, not from `spec.target`.** Postgres carries user and
password inside its DSN because libpq's own connection-string format has slots for them. Oracle's
does not: an Easy Connect descriptor is `host:port/service` and a TNS alias is a name, and neither
has anywhere to put a credential. Inventing an `oracle://user:pw@host/svc` URL to fill the gap
would mean percent-encoding rules that Oracle passwords break constantly -- `#` reads as a URL
fragment and `/` ends the netloc, so a password containing either would parse into a wrong DSN and
a wrong user with no error at all. Each database keeps its own native format, and Oracle's
credentials travel in `MNEMIQ_ORACLE_USER` / `MNEMIQ_ORACLE_PASSWORD`.

The consequence, stated rather than hidden: **v1 supports one Oracle source**, because one pair of
env vars can only describe one. A second would need per-source credential references, which is a
manifest-format change and not a v1 requirement.
"""

from __future__ import annotations

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.adapters.oracle import OracleAdapter
from mnemiq.config import Settings, SourceSpec

# The kinds a source may declare. `federated.py` supports a SUBSET of these -- see `FEDERABLE`.
KINDS = ("postgres", "sqlite", "duckdb", "oracle")

# DuckDB ships ATTACH scanners for PostgreSQL, MySQL and SQLite, and reads its own format. It has
# no Oracle scanner, so an Oracle source cannot join a federated query in v1. That is a refusal,
# not a silent omission: `FederatedAdapter` used to raise a bare `KeyError('oracle')` from an
# extension-table lookup, which tells an operator nothing about why their manifest is impossible.
FEDERABLE = ("postgres", "sqlite")

# The env var that configures a source of this kind when there is no manifest. Only Postgres has
# one: `Settings.source_specs` synthesizes its single legacy spec from `pg_dsn`, so an unconfigured
# engine yields a spec with an EMPTY target rather than no spec at all, and the operator needs the
# variable's name to act on the error. Naming only the source id -- which is `acme` by default and
# was never chosen by anyone -- tells them nothing they can do.
_TARGET_ENV = {"postgres": "MNEMIQ_PG_DSN"}

_ATTACH = {
    # kind -> (ATTACH type, DuckDB extension, default schema, fk lookup via postgres catalog)
    "postgres": ("POSTGRES", "postgres", "public", True),
    "sqlite": ("SQLITE", "sqlite", "main", False),
    "duckdb": ("DUCKDB", "", "main", False),
}


class UnknownSourceKind(ValueError):
    """A source declares a `kind` no adapter implements."""


class SourceUnconfigured(RuntimeError):
    """A source is declared but something it needs to connect is missing.

    Deliberately NOT a subclass of `runtime.SnapshotMissing`: that would make this module import
    `runtime`, which imports this one. `build_runtime` translates it at the call site instead, so
    the established `SnapshotMissing` contract for a missing source survives unchanged.
    """


def adapter_for(spec: SourceSpec, settings: Settings | None = None, *, read_only: bool = True):
    """The adapter for one source. `settings` is required only for Oracle's credentials.

    `read_only` is threaded through to every adapter rather than defaulted per-kind, because the
    read plane must not depend on the decider being the only thing between it and a write -- the
    finding **M3** is what happens when that switch reaches some adapters and not others.
    """
    if spec.kind not in KINDS:
        raise UnknownSourceKind(
            f"source {spec.id!r} declares kind={spec.kind!r}; known kinds: {', '.join(KINDS)}"
        )
    if not spec.target:
        env = _TARGET_ENV.get(spec.kind)
        raise SourceUnconfigured(
            f"source {spec.id!r} (kind={spec.kind}) has no target"
            + (f" -- set {env}" if env else "")
            + ", or give it one in the manifest at MNEMIQ_SOURCES_PATH"
        )

    if spec.kind == "oracle":
        user = (settings.oracle_user if settings else None) or None
        password = (settings.oracle_password if settings else None) or None
        missing = [n for n, v in (("MNEMIQ_ORACLE_USER", user),
                                  ("MNEMIQ_ORACLE_PASSWORD", password)) if not v]
        if missing:
            raise SourceUnconfigured(
                f"source {spec.id!r} is Oracle and needs {' and '.join(missing)} -- an Easy "
                "Connect descriptor carries no credentials, so they are configured separately"
            )
        return OracleAdapter(dsn=spec.target, user=user, password=password,
                             schema=spec.schema or None, read_only=read_only)

    attach_type, extension, default_schema, fk_via_postgres = _ATTACH[spec.kind]
    return DuckDBAdapter(
        attach_target=spec.target,
        attach_type=attach_type,
        extension=extension,
        catalog=spec.catalog or "src",
        table_schema=spec.schema or default_schema,
        fk_via_postgres=fk_via_postgres,
        read_only=read_only,
    )


def source_spec(settings: Settings, source_id: str | None = None) -> SourceSpec:
    """The one source a single-source command (`enrich`, `refresh`) operates on.

    Those commands took `settings.pg_dsn` and labelled the result `settings.source_id`, so against
    a multi-source manifest they enriched whatever Postgres was configured and filed it under a
    name that need not have described it. Selecting by id and refusing an ambiguous match is the
    smaller claim and the checkable one; it does NOT enrich every source in a manifest, which
    would be a new behaviour rather than a corrected one.
    """
    specs = settings.source_specs()
    if not specs:
        raise SourceUnconfigured("no sources configured -- set MNEMIQ_PG_DSN or MNEMIQ_SOURCES_PATH")
    if len(specs) == 1:
        return specs[0]
    wanted = source_id or settings.source_id
    match = [s for s in specs if s.id == wanted]
    if len(match) == 1:
        return match[0]
    raise SourceUnconfigured(
        f"{len(specs)} sources are configured and MNEMIQ_SOURCE_ID={wanted!r} "
        f"{'is ambiguous' if match else 'names none of them'}; "
        f"available: {', '.join(sorted(s.id for s in specs))}"
    )
