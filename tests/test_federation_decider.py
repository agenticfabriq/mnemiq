import sqlglot

from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.cls import check_cls
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import apply_row_and_mask
from mnemiq.sql.verdict import Approved, RefusalCode


def _ast(sql):
    return sqlglot.parse_one(sql, read="duckdb")


def test_check_access_accepts_qualified_and_refuses_bare_in_federated():
    visible = {"pg.person": {"id", "last_name"}, "ops.orders": {"id", "pid"}}
    assert check_access(_ast("SELECT id FROM pg.person"), visible) is None
    r = check_access(_ast("SELECT id FROM person"), visible)  # bare -> unknown in federated
    assert r is not None and r.code == RefusalCode.UNAUTHORIZED_TABLE


def test_check_access_qualified_cross_source_join():
    visible = {"pg.person": {"id", "last_name"}, "ops.orders": {"pid", "amount"}}
    sql = "SELECT p.last_name, o.amount FROM pg.person p JOIN ops.orders o ON o.pid = p.id"
    assert check_access(_ast(sql), visible) is None


def test_cls_denies_qualified_column():
    policy = AccessPolicy(denied={("pg.person", "last_name")})
    r = check_cls(_ast("SELECT last_name FROM pg.person"), policy)
    assert r is not None and r.code == RefusalCode.UNAUTHORIZED_COLUMN


def test_rls_wraps_qualified_table():
    policy = AccessPolicy(row_filters={"pg.person": "id > 0"})
    visible = {"pg.person": {"id", "last_name"}}
    out = apply_row_and_mask(_ast("SELECT id FROM pg.person"), policy, visible, dialect="duckdb")
    low = out.sql(dialect="duckdb").lower()
    assert "id > 0" in low and "from pg.person" in low  # filter applied at the (still-qualified) source


def test_decide_expands_qualified_table_for_execution():
    visible = {"pg.person": {"id", "last_name"}}
    v = decide("SELECT id FROM pg.person", visible, dialect="duckdb", target="duckdb",
               registry={"pg": "public"})
    assert isinstance(v, Approved)
    assert "pg.public.person" in v.target_sql
    assert v.tables == ["pg.person"]  # provenance keeps the qualified id


def test_decide_single_source_unchanged():
    visible = {"person": {"id"}}
    v = decide("SELECT id FROM person", visible, dialect="duckdb", target="duckdb")
    assert isinstance(v, Approved)
    assert v.target_sql.strip().lower().startswith("select") and "person" in v.target_sql
    assert v.tables == ["person"]
