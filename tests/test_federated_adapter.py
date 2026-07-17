import sqlite3

from mnemiq.adapters.federated import FederatedAdapter
from mnemiq.config import SourceSpec


def _sqlite(path, ddl_rows):
    con = sqlite3.connect(path)
    for stmt in ddl_rows:
        con.execute(stmt)
    con.commit()
    con.close()


def test_federated_cross_catalog_join(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    _sqlite(a, ["CREATE TABLE person (id INTEGER, name TEXT)",
                "INSERT INTO person VALUES (1,'Ann'),(2,'Bo')"])
    _sqlite(b, ["CREATE TABLE orders (pid INTEGER, amount INTEGER)",
                "INSERT INTO orders VALUES (1,10),(1,5),(2,7)"])
    specs = [
        SourceSpec(id="pa", kind="sqlite", target=str(a), catalog="pa", schema="main"),
        SourceSpec(id="pb", kind="sqlite", target=str(b), catalog="pb", schema="main"),
    ]
    adapter = FederatedAdapter(specs)
    assert adapter.dialect == "duckdb"
    assert adapter.registry == {"pa": "main", "pb": "main"}
    rows = adapter.execute(
        "SELECT p.name, sum(o.amount) FROM pa.main.person p "
        "JOIN pb.main.orders o ON o.pid = p.id GROUP BY p.name ORDER BY p.name"
    )
    assert rows == [("Ann", 15), ("Bo", 7)]
