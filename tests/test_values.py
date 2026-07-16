import duckdb

from mnemiq.contract import Column, Snapshot
from mnemiq.semantic.values import ValueIndex, build_value_index


class _Adapter:
    """Runs SQL against a throwaway in-memory DuckDB standing in for the source."""

    def __init__(self, con):
        self._con = con

    def execute(self, sql):
        return self._con.execute(sql).fetchall()


def _source():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE country (region TEXT, country_id TEXT, name TEXT, bio TEXT)")
    con.execute(
        "INSERT INTO country VALUES "
        "('EU','1','Alice','a'),('EU','2','Bob','b'),('APAC','3','Cara','c')"
    )
    return con


def _snapshot(columns):
    return Snapshot(
        version="v1", source_id="s", created_at="2026-07-15T00:00:00Z", columns=columns
    )


def _columns():
    return [
        Column(id="country.region", object_id="country", name="region",
               data_type="VARCHAR", distinct_count=2),
        Column(id="country.country_id", object_id="country", name="country_id",
               data_type="VARCHAR", distinct_count=3),  # key-like -> skip
        Column(id="country.name", object_id="country", name="name",
               data_type="VARCHAR", distinct_count=3),  # sensitive -> skip
        Column(id="country.bio", object_id="country", name="bio",
               data_type="VARCHAR", distinct_count=300),  # over max_distinct -> skip
        Column(id="country.rank", object_id="country", name="rank",
               data_type="INTEGER", distinct_count=3),  # non-string -> skip
    ]


def test_build_value_index_indexes_only_bounded_non_key_non_pii_string_columns():
    store = duckdb.connect(":memory:")
    n = build_value_index(_Adapter(_source()), _snapshot(_columns()), store)

    assert n == 2  # only 'region', two distinct values
    vi = ValueIndex(store)
    assert vi.has("country", "region")
    assert not vi.has("country", "country_id")  # key-like
    assert not vi.has("country", "name")  # sensitive
    assert not vi.has("country", "bio")  # too many distinct
    assert not vi.has("country", "rank")  # non-string
    assert vi.contains("country", "region", "EU")
    assert not vi.contains("country", "region", "emea")


def test_build_value_index_is_idempotent_per_source():
    store = duckdb.connect(":memory:")
    build_value_index(_Adapter(_source()), _snapshot(_columns()), store)
    n = build_value_index(_Adapter(_source()), _snapshot(_columns()), store)  # rebuild
    assert n == 2  # delete-then-insert: no duplication
    rows = store.execute("SELECT count(*) FROM value_index").fetchone()[0]
    assert rows == 2


def test_nearest_surfaces_the_intended_value_first():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE gasstations (Country TEXT)")
    con.execute(
        "INSERT INTO gasstations VALUES "
        "('Czech Republic'),('Slovakia'),('Poland'),('Austria'),('Germany')"
    )
    cols = [Column(id="gasstations.Country", object_id="gasstations", name="Country",
                   data_type="TEXT", distinct_count=5)]
    store = duckdb.connect(":memory:")
    build_value_index(_Adapter(con), _snapshot(cols), store)

    vi = ValueIndex(store)
    near = vi.nearest("gasstations", "Country", "CZE", k=8)
    assert near[0] == "Czech Republic"  # trigram overlap on 'cze' ranks it first
    assert len(near) == 5  # <= k -> all returned, ranked
