"""The ACME Postgres DSN, in one place.

A module rather than a conftest helper, and not because conftest would not work -- measured,
`from conftest import acme_dsn` resolves here, since pytest puts the test directory on
`sys.path`. It is a module because an import naming the thing it imports is easier to follow
than one naming the file pytest happens to load first, and because `requires_acme` below is a
plain value rather than a fixture and has no reason to live in a fixture file.
"""

import os

import pytest


def acme_dsn() -> str:
    """The ACME Postgres DSN, decided in ONE place and mirroring `docker-compose.yml`.

    Fifteen occurrences across fifteen files, enumerated rather than counted: thirteen
    spelling port 5433 and two spelling 5432. **5433 is not this project's port** --
    `docker-compose.yml` declares `${MNEMIQ_PG_PORT:-5432}`. It is one machine, where another
    project's Postgres took 5432 first and the operator recorded the override in `.env`.

    The engine reads `os.environ` only (`Settings` says so), so a bare `pytest` never sees
    that file and every module fell through to its literal: the thirteen on 5433 passed by
    coincidence, and the two on 5432 reached the OTHER project's database and failed
    authentication. Fixing THAT by moving the two to 5433 was the wrong direction, and a
    review caught it -- it takes one machine's port collision for the project's truth.

    Compose's own precedence: an explicit DSN, else a port override, else compose's declared
    default.

    ONLY THE FIRST LEVEL DECIDES ANYTHING A TEST SEES. `requires_acme` skips unless
    MNEMIQ_PG_DSN is set, so a test that actually runs always took that branch. The other two
    exist because `_DSN = acme_dsn()` sits at MODULE level in thirteen of the fifteen callers
    and is evaluated at import, before a skip mark can apply -- so this has to return a string
    rather than raise, or collection fails on a machine with no database. (The two exceptions
    call it lazily inside a `_dsn()` helper, and they are exactly the two that historically
    spelled 5432.) They document the project's port; they are not a working fallback, and
    reading them as one is how this docstring first described them.
    """
    dsn = os.getenv("MNEMIQ_PG_DSN")
    if dsn:
        return dsn
    port = os.getenv("MNEMIQ_PG_PORT", "5432")
    return f"postgresql://mnemiq:mnemiq@localhost:{port}/acme"


# EXPLICIT OPT-IN, matching what this project already does everywhere else it needs a live
# database: `test_write_live` skips without MNEMIQ_PG_DSN, `test_oracle_adapter` skips without
# MNEMIQ_ORACLE_TEST_DSN, and `conftest._seed_acme` no-ops without both its variables.
#
# The alternative -- try the compose default and let the connection fail -- is what these
# modules did, and it produced seventeen red tests on a machine whose Postgres is not on the
# default port. A red test means a defect; "no database here" is not one. Skipping says so,
# and pytest prints the count, so the coverage is not silently lost either.
requires_acme = pytest.mark.skipif(
    not os.getenv("MNEMIQ_PG_DSN"),
    reason="set MNEMIQ_PG_DSN to run the ACME Postgres tests (see .env / docker-compose.yml)",
)


# ACME has 29 tables and still does. The catalogue reports 32 because the ENGINE writes three
# of its own into the same schema -- mnemiq_answer_log, mnemiq_eval_run, mnemiq_version -- so
# they are introspected and reach the snapshot. `== 29` did not break because the dataset
# grew, which is what the first version of this fix claimed. That the engine's tables are
# introspected at all is a question about the engine, recorded rather than changed here.
ACME_MIN_TABLES = 29

# `'mnemiq%'`, NOT `'mnemiq\_%'`. Measured through this adapter: the underscore form returns
# 32 and the plain form 29, so the backslash was read as a literal and the exclusion silently
# did not happen. A backslash is not a LIKE escape without an ESCAPE clause here; whether any
# given engine treats it as one is exactly the assumption that made this wrong, so the
# predicate avoids needing to know. Nothing in ACME begins with "mnemiq".
_NOT_ENGINE = "table_name NOT LIKE 'mnemiq%'"
_PUBLIC = "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"


def source_table_count(adapter) -> int:
    """Every table in the attached catalogue's `public` schema, engine bookkeeping included.

    SINGLE-ATTACH ONLY. The adapter's own reads also filter `table_catalog`, and this does
    not: `DuckDBPostgresAdapter` attaches exactly one catalog so the two agree, but
    `FederatedAdapter` attaches one per source into one connection, where this would sum
    them. Every caller here is the single-source adapter; a federated caller would need the
    catalogue name passed in.
    """
    return adapter.execute(_PUBLIC)[0][0]


def assert_acme_seeded(adapter) -> None:
    """A FLOOR on ACME's own tables, which asking the source alone cannot give.

    Comparing a snapshot to the catalogue only proves they agree: a half-seeded ACME agrees
    with itself and passes. The floor is what validates the FIXTURE, and it must EXCLUDE the
    engine's three tables -- counting all 32 pads it by exactly three, so a fixture missing
    three ACME tables would still clear a floor of 29, absorbing the case it exists to catch.
    """
    n = adapter.execute(f"{_PUBLIC} AND {_NOT_ENGINE}")[0][0]
    assert n >= ACME_MIN_TABLES, f"ACME looks unseeded: {n} of its tables present"
