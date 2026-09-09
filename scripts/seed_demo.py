"""Create the demo database the quickstart runs against — SQLite, stdlib only, no Docker.

The quickstart used to end at `export MNEMIQ_PG_DSN=...`, which asks a first-time reader to
supply the one thing they do not have. This writes a small database, a source manifest and an
authorization policy under `demo/`, so the four commands after it work on a clean clone with
nothing installed but the package itself.

The data is invented and deliberately small: four tables, a few dozen rows, enough for joins
and aggregates without making enrichment slow. It is also deliberately INCOMPLETE — there is
no revenue, price or attendance column anywhere — so that asking for revenue demonstrates the
engine stating what it cannot answer rather than inventing a number.

    uv run python scripts/seed_demo.py
"""

from __future__ import annotations

import json
import os
import sqlite3

DEMO_DIR = "demo"
DB_PATH = os.path.join(DEMO_DIR, "demo.sqlite")

SCHEMA = """
CREATE TABLE customer (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    country       TEXT    NOT NULL,
    signup_date   TEXT    NOT NULL,
    segment_cd    TEXT    NOT NULL   -- coded: A, B, C. Meaning lives in segment_lookup.
);
CREATE TABLE segment_lookup (
    segment_cd    TEXT PRIMARY KEY,
    segment_name  TEXT NOT NULL
);
CREATE TABLE product (
    product_id    INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    category      TEXT    NOT NULL
);
CREATE TABLE order_line (
    order_id      INTEGER NOT NULL,
    customer_id   INTEGER NOT NULL REFERENCES customer(customer_id),
    product_id    INTEGER NOT NULL REFERENCES product(product_id),
    quantity      INTEGER NOT NULL,
    order_date    TEXT    NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
"""

CUSTOMERS = [
    (1, "Aurora Bakehouse", "France", "2024-01-14", "A"),
    (2, "Meridian Foods", "France", "2024-02-02", "B"),
    (3, "Northwind Grocers", "Netherlands", "2024-02-19", "A"),
    (4, "Perch & Co", "United States", "2024-03-07", "C"),
    (5, "Saltmarsh Trading", "France", "2024-04-21", "B"),
    (6, "Vellum Supply", "United States", "2024-05-30", "A"),
    (7, "Harrow Provisions", "Germany", "2024-06-11", "C"),
    (8, "Cinder Lane Cafe", "Netherlands", "2024-07-03", "B"),
]
SEGMENTS = [("A", "Enterprise"), ("B", "Mid-market"), ("C", "Small business")]
PRODUCTS = [
    (1, "Stone-ground flour", "Dry goods"),
    (2, "Cultured butter", "Dairy"),
    (3, "Sourdough starter", "Dry goods"),
    (4, "Single-origin cocoa", "Dry goods"),
    (5, "Crème fraîche", "Dairy"),
    (6, "Sea salt flakes", "Pantry"),
]
ORDER_LINES = [
    (1001, 1, 1, 12, "2024-08-02"), (1001, 1, 2, 4, "2024-08-02"),
    (1002, 3, 1, 30, "2024-08-05"), (1002, 3, 6, 8, "2024-08-05"),
    (1003, 2, 4, 6, "2024-08-11"), (1004, 5, 2, 15, "2024-08-14"),
    (1004, 5, 5, 9, "2024-08-14"), (1005, 6, 3, 3, "2024-08-19"),
    (1006, 4, 1, 22, "2024-09-01"), (1006, 4, 4, 2, "2024-09-01"),
    (1007, 8, 6, 40, "2024-09-04"), (1008, 1, 3, 7, "2024-09-09"),
    (1009, 7, 2, 11, "2024-09-15"), (1010, 3, 5, 5, "2024-09-21"),
    (1010, 3, 2, 18, "2024-09-21"), (1011, 6, 6, 25, "2024-10-02"),
]

SOURCES = [
    {
        "id": "demo",
        "kind": "sqlite",
        # Relative on purpose: an absolute path here is one machine's path, and the manifest
        # is the file a reader is most likely to copy into their own project.
        "target": DB_PATH,
        "catalog": "src",
        "schema": "main",
    }
]
AUTHZ = {
    "roles": {
        "analyst": ["customer", "segment_lookup", "product", "order_line"],
    }
}


def main() -> int:
    os.makedirs(DEMO_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO customer VALUES (?,?,?,?,?)", CUSTOMERS)
        con.executemany("INSERT INTO segment_lookup VALUES (?,?)", SEGMENTS)
        con.executemany("INSERT INTO product VALUES (?,?,?)", PRODUCTS)
        con.executemany("INSERT INTO order_line VALUES (?,?,?,?,?)", ORDER_LINES)
        con.commit()
    finally:
        con.close()

    with open(os.path.join(DEMO_DIR, "sources.json"), "w") as fh:
        json.dump(SOURCES, fh, indent=2)
        fh.write("\n")
    with open(os.path.join(DEMO_DIR, "authz.json"), "w") as fh:
        json.dump(AUTHZ, fh, indent=2)
        fh.write("\n")

    rows = sum(len(t) for t in (CUSTOMERS, SEGMENTS, PRODUCTS, ORDER_LINES))
    print(f"wrote {DB_PATH} — 4 tables, {rows} rows")
    print(f"wrote {DEMO_DIR}/sources.json, {DEMO_DIR}/authz.json")
    print()
    print("Next:")
    print("  export MNEMIQ_SOURCES_PATH=demo/sources.json")
    print("  export MNEMIQ_AUTHZ_PATH=demo/authz.json")
    print("  export MNEMIQ_STORE_PATH=demo/store.duckdb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
