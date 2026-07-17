import sqlglot

from mnemiq.sql.qualify import expand_tables, object_key


def _table(sql):
    return sqlglot.parse_one(sql, read="duckdb").find(sqlglot.exp.Table)


def test_object_key_bare_is_name():
    assert object_key(_table("SELECT * FROM person")) == "person"


def test_object_key_qualified_is_catalog_dot_table():
    assert object_key(_table("SELECT * FROM pg.person")) == "pg.person"


def test_expand_tables_moves_catalog_and_inserts_schema():
    ast = sqlglot.parse_one("SELECT a FROM pg.person", read="duckdb")
    expand_tables(ast, {"pg": "public"})
    assert ast.sql(dialect="duckdb") == "SELECT a FROM pg.public.person"


def test_expand_tables_empty_registry_is_noop():
    ast = sqlglot.parse_one("SELECT a FROM person", read="duckdb")
    before = ast.sql(dialect="duckdb")
    expand_tables(ast, {})
    assert ast.sql(dialect="duckdb") == before


def test_expand_tables_leaves_unknown_catalog_alone():
    ast = sqlglot.parse_one("SELECT a FROM other.person", read="duckdb")
    expand_tables(ast, {"pg": "public"})
    assert ast.sql(dialect="duckdb") == "SELECT a FROM other.person"
