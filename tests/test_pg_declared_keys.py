"""The declared foreign keys a direct Postgres source reports, read from a real Postgres.

Each test reads keys from a database this module makes for itself on the server MNEMIQ_TEST_PG_KEYS_DSN names, and
drops when it is done: a single-column key, a composite key declared in another column order than its tables', two keys
between one pair of tables, two tables each declaring a key of one name, a key into another schema's table whose name
`public` has too, a partitioned table that declares a key and is referenced by one, a second schema, `sales`, with keys
of its own, a schema stored in mixed case, and two schemas whose names differ only in letter case. Without the variable
each of those tests skips, naming it, or fails where MNEMIQ_TEST_PG_KEYS_REQUIRED is `1`, as in the CI job that runs
them against a Postgres of its own. The tests of that switch need no server.

Not marked `integration`: the root conftest seeds ACME into MNEMIQ_PG_DSN whenever an integration test is collected and
both MNEMIQ_PG_DSN and MNEMIQ_ACME_DATA_DIR are set, and these tests need nothing of ACME's.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from mnemiq.adapters.resolve import adapter_for
from mnemiq.catalog import introspect
from mnemiq.config import SourceSpec
from mnemiq.enrichment.joins import relationships_from_foreign_keys

DSN_VARIABLE = "MNEMIQ_TEST_PG_KEYS_DSN"
REQUIRED_VARIABLE = "MNEMIQ_TEST_PG_KEYS_REQUIRED"

# The objects each test database holds. Every key is named, so a test can say which constraint a row belongs to.
_DDL = """
CREATE TABLE customers (id int PRIMARY KEY);
CREATE TABLE orders (id int PRIMARY KEY, customer_id int,
    CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES customers (id));

-- A composite key declared (b_id, a_id), against tables whose columns are in another order.
CREATE TABLE pair (a_id int, b_id int, PRIMARY KEY (a_id, b_id));
CREATE TABLE pair_line (id int PRIMARY KEY, a_id int, b_id int,
    CONSTRAINT line_pair_fk FOREIGN KEY (b_id, a_id) REFERENCES pair (b_id, a_id));

-- Two keys between one pair of tables.
CREATE TABLE account (id int PRIMARY KEY);
CREATE TABLE transfer (id int PRIMARY KEY, from_account int, to_account int,
    CONSTRAINT transfer_from_fk FOREIGN KEY (from_account) REFERENCES account (id),
    CONSTRAINT transfer_to_fk FOREIGN KEY (to_account) REFERENCES account (id));

-- Two tables each declaring a key named `fk_customer`: Postgres keeps a key's name unique on its table, not in its
-- schema.
CREATE TABLE invoices (id int PRIMARY KEY, cust int,
    CONSTRAINT fk_customer FOREIGN KEY (cust) REFERENCES customers (id));
CREATE TABLE payments (id int PRIMARY KEY, customer_id int,
    CONSTRAINT fk_customer FOREIGN KEY (customer_id) REFERENCES customers (id));

-- A key into schema `other`, whose `region` has a namesake in `public`, beside a key into that namesake.
CREATE SCHEMA other;
CREATE TABLE other.region (id int PRIMARY KEY);
CREATE TABLE region (id int PRIMARY KEY);
CREATE TABLE store (id int PRIMARY KEY, region_id int, home_region_id int,
    CONSTRAINT store_other_region_fk FOREIGN KEY (region_id) REFERENCES other.region (id),
    CONSTRAINT store_home_region_fk FOREIGN KEY (home_region_id) REFERENCES region (id));

-- A partitioned table that declares a key and that a key references. Postgres keeps a copy of each key for each
-- partition, naming the partition in place of the table.
CREATE TABLE warehouse (id int PRIMARY KEY);
CREATE TABLE event (id int, region_id int, warehouse_id int, PRIMARY KEY (id, region_id),
    CONSTRAINT event_warehouse_fk FOREIGN KEY (warehouse_id) REFERENCES warehouse (id)) PARTITION BY LIST (region_id);
CREATE TABLE event_r1 PARTITION OF event FOR VALUES IN (1);
CREATE TABLE event_r2 PARTITION OF event FOR VALUES IN (2);
CREATE TABLE event_note (id int PRIMARY KEY, event_id int, event_region int,
    CONSTRAINT note_event_fk FOREIGN KEY (event_id, event_region) REFERENCES event (id, region_id));

-- A second schema with keys of its own, a table named as one in `public`, and a key into `public`.
CREATE SCHEMA sales;
CREATE TABLE sales.region (id int PRIMARY KEY);
CREATE TABLE sales.deal (id int PRIMARY KEY, region_id int, customer_id int,
    CONSTRAINT deal_region_fk FOREIGN KEY (region_id) REFERENCES sales.region (id),
    CONSTRAINT deal_customer_fk FOREIGN KEY (customer_id) REFERENCES public.customers (id));

-- A schema stored in mixed case, and two schemas whose names differ only in letter case, each with a key.
CREATE SCHEMA "Archive";
CREATE TABLE "Archive".box (id int PRIMARY KEY);
CREATE TABLE "Archive".item (id int PRIMARY KEY, box_id int,
    CONSTRAINT item_box_fk FOREIGN KEY (box_id) REFERENCES "Archive".box (id));
CREATE SCHEMA "Twin";
CREATE TABLE "Twin".upper_parent (id int PRIMARY KEY);
CREATE TABLE "Twin".upper_child (id int PRIMARY KEY, parent_id int,
    CONSTRAINT upper_child_fk FOREIGN KEY (parent_id) REFERENCES "Twin".upper_parent (id));
CREATE SCHEMA twin;
CREATE TABLE twin.lower_parent (id int PRIMARY KEY);
CREATE TABLE twin.lower_child (id int PRIMARY KEY, parent_id int,
    CONSTRAINT lower_child_fk FOREIGN KEY (parent_id) REFERENCES twin.lower_parent (id));
"""

# Every declared key in `public` whose two ends are both in `public`, one row for each column pair, in the order the
# statement reads them: by table, then position in the key, then the key's name.
PUBLIC_KEYS = [
    ("event", "warehouse_id", "warehouse", "id", "event_warehouse_fk"),
    ("event_note", "event_id", "event", "id", "note_event_fk"),
    ("event_note", "event_region", "event", "region_id", "note_event_fk"),
    ("invoices", "cust", "customers", "id", "fk_customer"),
    ("orders", "customer_id", "customers", "id", "orders_customer_fk"),
    ("pair_line", "b_id", "pair", "b_id", "line_pair_fk"),
    ("pair_line", "a_id", "pair", "a_id", "line_pair_fk"),
    ("payments", "customer_id", "customers", "id", "fk_customer"),
    ("store", "home_region_id", "region", "id", "store_home_region_fk"),
    ("transfer", "from_account", "account", "id", "transfer_from_fk"),
    ("transfer", "to_account", "account", "id", "transfer_to_fk"),
]
SALES_KEYS = [("deal", "region_id", "region", "id", "deal_region_fk")]


def _server_dsn() -> str:
    """The Postgres server the tests make their database on, from MNEMIQ_TEST_PG_KEYS_DSN. Without it they skip, unless
    MNEMIQ_TEST_PG_KEYS_REQUIRED is `1`, as in the CI job that gives them a server: there they fail instead, so that job
    cannot pass by skipping them. A switch set to anything but `1` or `0` fails too, rather than reading as unset."""
    required = os.environ.get(REQUIRED_VARIABLE, "").strip()
    if required not in ("", "0", "1"):
        pytest.fail(f"{REQUIRED_VARIABLE} is {required!r}: set it to 1 to make these tests fail rather than skip "
                    f"without {DSN_VARIABLE}, or to 0 or nothing to let them skip", pytrace=False)
    dsn = os.environ.get(DSN_VARIABLE, "").strip()
    if not dsn:
        reason = f"{DSN_VARIABLE} is not set: these tests need a Postgres server to make their own database on"
        if required == "1":
            pytest.fail(f"{reason}, and {REQUIRED_VARIABLE}=1 says they must run here, not skip", pytrace=False)
        pytest.skip(reason)
    return dsn


def _make_database(server: str, ddl: str) -> tuple[str, str]:
    """A new database on `server` holding `ddl`'s objects: its name and a DSN for it."""
    name = f"mnemiq_keys_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(server, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    dsn = make_conninfo(server, dbname=name)
    try:
        with psycopg.connect(dsn, autocommit=True) as con:
            con.execute(ddl)
    except BaseException:
        _drop_database(server, name)
        raise
    return name, dsn


def _drop_database(server: str, name: str) -> None:
    with psycopg.connect(server, autocommit=True) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture(scope="module")
def keys_dsn() -> Iterator[str]:
    """A DSN for a database holding `_DDL`'s objects, made for this module and dropped after it."""
    server = _server_dsn()
    name, dsn = _make_database(server, _DDL)
    try:
        yield dsn
    finally:
        _drop_database(server, name)


@pytest.fixture
def no_superset_mode(monkeypatch) -> None:
    """`adapter_for` refuses a direct source when the environment sets MNEMIQ_SUPERSET_MCP_URL, in any case."""
    for name in list(os.environ):
        if name.upper() == "MNEMIQ_SUPERSET_MCP_URL":
            monkeypatch.delenv(name)


def _adapter(dsn: str, schema: str = ""):
    """A Postgres source on `dsn`, built as a source spec builds it: `schema` is the spec's schema, which the resolver
    reads as `public` when empty."""
    return adapter_for(SourceSpec(id="keys", kind="postgres", target=dsn, catalog="src", schema=schema))


def _keys(dsn: str, schema: str = "") -> list[tuple[str, str, str, str, str]]:
    """The keys that source reports."""
    return _adapter(dsn, schema).foreign_keys()


def _of(keys: list[tuple], *tables: str) -> list[tuple]:
    """The rows of `keys` whose source or target table is one of `tables`."""
    return [key for key in keys if key[0] in tables or key[2] in tables]


# The gate itself, with no server. Its outcome is caught rather than let through: a test of whether the gate skips must
# not be one the gate can skip. The switch is spelled as the CI job that runs this module sets it.
SWITCH = "MNEMIQ_TEST_PG_KEYS_REQUIRED"


def _gate(monkeypatch, dsn: str | None, switch: str | None) -> tuple[str, str]:
    """What `_server_dsn` does with `dsn` and `switch` as the environment's values (None: unset): `returned`, `skipped`
    or `failed`, with the DSN it returned or the reason it gave."""
    for name, value in ((DSN_VARIABLE, dsn), (SWITCH, switch)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    try:
        return "returned", _server_dsn()
    except pytest.skip.Exception as exc:
        return "skipped", str(exc)
    except pytest.fail.Exception as exc:
        return "failed", str(exc)


@pytest.mark.parametrize("switch", [None, "", "0"])
def test_without_the_dsn_and_without_the_switch_the_tests_skip_naming_the_dsn(monkeypatch, switch):
    outcome, reason = _gate(monkeypatch, None, switch)
    assert outcome == "skipped", reason
    assert DSN_VARIABLE in reason


def test_with_the_switch_and_without_the_dsn_the_tests_fail_naming_both(monkeypatch):
    """The CI job that runs this module sets the switch, so it cannot pass by skipping every test here."""
    outcome, reason = _gate(monkeypatch, None, "1")
    assert outcome == "failed", reason
    assert DSN_VARIABLE in reason and SWITCH in reason, reason


@pytest.mark.parametrize("switch", [None, "1"])
def test_with_the_dsn_the_tests_run_whatever_the_switch(monkeypatch, switch):
    assert _gate(monkeypatch, "postgresql://keys@127.0.0.1/postgres", switch) == (
        "returned", "postgresql://keys@127.0.0.1/postgres")


@pytest.mark.parametrize("dsn", [None, "postgresql://keys@127.0.0.1/postgres"])
@pytest.mark.parametrize("switch", ["true", "yes", "on"])
def test_a_switch_set_to_anything_but_1_or_0_fails_naming_it(monkeypatch, dsn, switch):
    """`true` is what an unquoted YAML `true` becomes in a workflow's environment. Read as unset, it would let the job
    pass by skipping; it fails instead, with or without the DSN, so the mistake shows on the run that makes it."""
    outcome, reason = _gate(monkeypatch, dsn, switch)
    assert outcome == "failed", reason
    assert SWITCH in reason and repr(switch) in reason, reason


# The workflow that gives these tests a server. Read as text: no YAML parser is a dependency of the `dev` extra.
WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "postgres-keys.yml"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _scalar(raw: str) -> str:
    """A plain or quoted YAML scalar's text, without a trailing comment."""
    quoted = re.match(r"""(["'])(.*?)\1""", raw)
    return quoted.group(2) if quoted else raw.split(" #", 1)[0].strip()


def _key_step_env(text: str) -> dict[str, str]:
    """The `env` of the one step in `text` whose `- run:` line runs pytest on this module, name to value. Read by
    indentation, as YAML nests it: the step is every line indented past its dash, its `env:` is the step's own key, and
    the variables are the lines indented past that. Blank lines and full-line comments are passed over."""
    lines = text.splitlines()
    module = f"tests/{Path(__file__).name}"
    starts = [i for i, line in enumerate(lines)
              if line.lstrip().startswith("- run:") and {"pytest", module} <= set(line.split())]
    assert len(starts) == 1, f"expected one step running pytest on {module} in {WORKFLOW.name}, found {len(starts)}"
    dash = _indent(lines[starts[0]])
    body = []
    for line in lines[starts[0] + 1:]:
        if line.strip() and not line.lstrip().startswith("#"):
            if _indent(line) <= dash:
                break
            body.append(line)
    envs = [i for i, line in enumerate(body) if line.strip() == "env:" and _indent(line) == dash + 2]
    assert len(envs) == 1, f"the step running {module} has no env of its own in {WORKFLOW.name}"
    env = {}
    for line in body[envs[0] + 1:]:
        if _indent(line) <= dash + 2:
            break
        name, _, value = line.strip().partition(":")
        env[name] = _scalar(value.strip())
    return env


def test_the_ci_workflow_sets_what_the_gate_reads():
    """The job that gives these tests a server sets the DSN and the switch under the names `_server_dsn` reads, the
    switch to `1`. A line dropped, or renamed on one side only, would leave the gate not reading what the job sets: the
    tests would fail there for want of a DSN set under another name, or, with the switch gone too, skip, and the job
    pass."""
    env = _key_step_env(WORKFLOW.read_text())
    assert env.get(DSN_VARIABLE), f"{WORKFLOW.name} does not set {DSN_VARIABLE} for the key tests: {env}"
    assert env.get(REQUIRED_VARIABLE) == "1", f"{WORKFLOW.name} does not set {REQUIRED_VARIABLE} to 1: {env}"


@pytest.mark.usefixtures("no_superset_mode")
def test_every_declared_key_in_public_and_nothing_else(keys_dsn):
    assert _keys(keys_dsn) == PUBLIC_KEYS


@pytest.mark.usefixtures("no_superset_mode")
def test_a_single_column_key(keys_dsn):
    assert _of(_keys(keys_dsn), "orders") == [("orders", "customer_id", "customers", "id", "orders_customer_fk")]


@pytest.mark.usefixtures("no_superset_mode")
def test_a_composite_key_pairs_each_column_with_its_own_target(keys_dsn):
    """Declared `(b_id, a_id) REFERENCES pair (b_id, a_id)`, where both tables hold `a_id` before `b_id`: two pairs,
    each source column with the target column at its own position in the key, under one constraint."""
    assert _of(_keys(keys_dsn), "pair_line") == [("pair_line", "b_id", "pair", "b_id", "line_pair_fk"),
                                                 ("pair_line", "a_id", "pair", "a_id", "line_pair_fk")]


@pytest.mark.usefixtures("no_superset_mode")
def test_a_key_into_another_schema_is_left_out(keys_dsn):
    """`store.region_id` references `other.region`, and `public` has a `region` too. A table here is named by its name
    alone, so the key is not listed at all rather than attached to `public.region`. The control: `store`'s key into
    `public.region` itself is listed."""
    keys = _keys(keys_dsn)
    assert _of(keys, "store") == [("store", "home_region_id", "region", "id", "store_home_region_fk")]
    assert [key for key in keys if key[4] == "store_other_region_fk"] == []


@pytest.mark.usefixtures("no_superset_mode")
def test_a_partitioned_table_lists_only_its_declared_keys(keys_dsn):
    """`event` is partitioned, declares a key and is referenced by one. Each key is listed once, as declared, and no
    partition's copy of either is."""
    assert _of(_keys(keys_dsn), "event", "event_note", "event_r1", "event_r2") == [
        ("event", "warehouse_id", "warehouse", "id", "event_warehouse_fk"),
        ("event_note", "event_id", "event", "id", "note_event_fk"),
        ("event_note", "event_region", "event", "region_id", "note_event_fk"),
    ]


@pytest.mark.usefixtures("no_superset_mode")
def test_two_keys_between_one_pair_of_tables_stay_two(keys_dsn):
    assert _of(_keys(keys_dsn), "transfer") == [("transfer", "from_account", "account", "id", "transfer_from_fk"),
                                                ("transfer", "to_account", "account", "id", "transfer_to_fk")]


@pytest.mark.usefixtures("no_superset_mode")
def test_two_tables_declaring_a_key_of_one_name_are_two_relationships(keys_dsn):
    """`invoices` and `payments` each declare `fk_customer`. Read through the adapter and made relationships by
    enrichment, each is a relationship of its own table, with its own column, beside `orders`'s key to the same table."""
    adapter = _adapter(keys_dsn)
    relationships = [r for r in relationships_from_foreign_keys(adapter, introspect(adapter)) if r.to == "customers"]
    assert [(r.id, r.from_, [(k.left, k.right) for k in r.join_keys]) for r in relationships] == [
        ("invoices.cust->customers", "invoices", [("cust", "id")]),
        ("orders.customer_id->customers", "orders", [("customer_id", "id")]),
        ("payments.customer_id->customers", "payments", [("customer_id", "id")]),
    ]


@pytest.mark.usefixtures("no_superset_mode")
def test_a_configured_schema_other_than_public_reads_its_own_keys(keys_dsn):
    """A source whose spec sets schema `sales` reads `sales`'s keys, none of `public`'s, and not its key into `public`;
    the default source reads none of `sales`'s."""
    assert _keys(keys_dsn, "sales") == SALES_KEYS
    assert not {"deal_region_fk", "deal_customer_fk"} & {key[4] for key in _keys(keys_dsn)}


@pytest.mark.usefixtures("no_superset_mode")
@pytest.mark.parametrize(("configured", "keys"), [
    ("SALES", SALES_KEYS),
    ("Sales", SALES_KEYS),
    ("archive", [("item", "box_id", "box", "id", "item_box_fk")]),
    ("ARCHIVE", [("item", "box_id", "box", "id", "item_box_fk")]),
])
def test_a_schema_configured_in_another_letter_case_reads_the_stored_schemas_keys(keys_dsn, configured, keys, caplog):
    """The adapter lists a schema's tables without regard to case, and so reads its keys: `SALES` reads the stored
    `sales`, `archive` the stored `Archive`, with nothing logged."""
    with caplog.at_level(logging.WARNING, logger="mnemiq.adapters.duckdb"):
        assert _keys(keys_dsn, configured) == keys
    assert [r for r in caplog.records if r.name == "mnemiq.adapters.duckdb"] == []


@pytest.mark.usefixtures("no_superset_mode")
@pytest.mark.parametrize("configured", ["twin", "Twin", "TWIN"])
def test_two_schemas_differing_only_in_letter_case_read_no_keys(keys_dsn, configured, caplog):
    """`Twin` and `twin` are both stored, and the configured name, matched without regard to case, names both, whether
    or not it spells one of them exactly. Which one is meant is not settled, so no key is read rather than one schema's
    keys given as the other's: `[]`, and one WARNING naming both schemas."""
    adapter = _adapter(keys_dsn, configured)
    with caplog.at_level(logging.WARNING, logger="mnemiq.adapters.duckdb"):
        assert adapter.foreign_keys() == []
    (record,) = [r for r in caplog.records if r.name == "mnemiq.adapters.duckdb"]
    assert record.levelno == logging.WARNING
    assert "schemas 'Twin' and 'twin' both match it without regard to case" in record.getMessage()


@pytest.mark.usefixtures("no_superset_mode")
def test_a_read_from_a_database_that_is_gone_logs_its_type_and_returns_nothing(caplog):
    """The database is dropped under an attached adapter. The read returns `[]`, as one whose keys enrichment then
    infers from the data, and logs one WARNING that names the exception's type, and none of its text, which can carry
    the DSN. The control: the same adapter read the key while the database was there."""
    server = _server_dsn()
    name, dsn = _make_database(server, "CREATE TABLE a (id int PRIMARY KEY); "
                                       "CREATE TABLE b (id int PRIMARY KEY, a_id int, "
                                       "CONSTRAINT b_a_fk FOREIGN KEY (a_id) REFERENCES a (id));")
    try:
        adapter = adapter_for(SourceSpec(id="gone", kind="postgres", target=dsn, catalog="src", schema=""))
        assert adapter.foreign_keys() == [("b", "a_id", "a", "id", "b_a_fk")]
    finally:
        _drop_database(server, name)
    with caplog.at_level(logging.WARNING, logger="mnemiq.adapters.duckdb"):
        assert adapter.foreign_keys() == []
    (record,) = [r for r in caplog.records if r.name == "mnemiq.adapters.duckdb"]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    # DuckDB raises an IOException whose text quotes the whole DSN it could not connect with.
    assert "(IOException)" in message, message
    parts = conninfo_to_dict(dsn)
    for secret_ish in (name, parts.get("host"), parts.get("port"), dsn):
        assert not secret_ish or str(secret_ish) not in message, message
