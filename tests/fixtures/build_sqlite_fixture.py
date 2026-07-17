import sqlite3


def build(path: str, pids: list[int]) -> None:
    """A tiny companion source: one order row per given person id, so a cross-source join to
    ACME's person table is exercised."""
    con = sqlite3.connect(path)
    con.execute("DROP TABLE IF EXISTS ext_orders")
    con.execute("CREATE TABLE ext_orders (pid INTEGER, amount INTEGER)")
    con.executemany("INSERT INTO ext_orders VALUES (?, ?)", [(p, p * 10) for p in pids])
    con.commit()
    con.close()
