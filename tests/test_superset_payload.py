"""How mnemiq reads what Superset's `execute_sql` tool answers: a result into an Arrow table typed by the pandas dtype
Superset reports for each column, and a failure into the exception that says whose it is to fix.

Every payload has the shape `execute_sql` sends (`ExecuteSqlResponse` in `superset/mcp_service/sql_lab/schemas.py`):
the rows as one JSON object each, and each column's name beside the pandas dtype it had in Superset's result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pytest

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.superset.client import SupersetUnavailable
from mnemiq.superset.payload import (
    ResultCut,
    SupersetAccessRefused,
    SupersetQueryFailed,
    read_execute_response,
    table_from_payload,
)

SUPERSET_TEXT = "Superset's own words about this query"


def _answer(rows: list[dict], columns: list[dict], **extra) -> dict:
    """A successful `execute_sql` payload, every field Superset sends beside the rows and columns included."""
    return {"success": True, "rows": rows, "columns": columns, "row_count": len(rows), "affected_rows": None,
            "execution_time": 0.012, "error": None, "error_type": None, "statements": [],
            "multi_statement_warning": None, "template_warning": None, **extra}


REGION_COUNTS = _answer([{"region": "west", "policy_count": 16666}],
                        [{"name": "region", "type": "object"}, {"name": "policy_count", "type": "int64"}])


def test_types_follow_the_reported_dtype():
    columns = [{"name": "region", "type": "object"},
               {"name": "policy_count", "type": "int64"},
               {"name": "claims", "type": "Int32"},
               {"name": "premium", "type": "float64"},
               {"name": "active", "type": "bool"},
               {"name": "lapsed", "type": "boolean"},
               {"name": "written_at", "type": "datetime64[ns]"},
               {"name": "paid_at", "type": "datetime64[ns, UTC]"},
               {"name": "booked_at", "type": "datetime64[us, +02:00]"}]
    # Each row's keys in another order than `columns`, which decides the table's order.
    rows = [{"booked_at": "2024-03-01T11:30:00+02:00", "paid_at": "2024-03-01T09:30:00Z",
             "written_at": "2024-03-01T09:30:00", "lapsed": False, "active": True, "premium": 1200.5, "claims": 3,
             "policy_count": 16666, "region": "west"},
            {"booked_at": None, "paid_at": None, "written_at": "2024-03-02T10:00:00.250000", "lapsed": None,
             "active": False, "premium": 2, "claims": None, "policy_count": 12, "region": "east"}]

    table = read_execute_response(_answer(rows, columns))

    assert table.column_names == [column["name"] for column in columns]
    assert table.schema.types == [pa.string(), pa.int64(), pa.int64(), pa.float64(), pa.bool_(), pa.bool_(),
                                  pa.timestamp("us"), pa.timestamp("us", tz="UTC"), pa.timestamp("us", tz="UTC")]
    assert table.to_pylist() == [
        {"region": "west", "policy_count": 16666, "claims": 3, "premium": 1200.5, "active": True, "lapsed": False,
         "written_at": datetime(2024, 3, 1, 9, 30), "paid_at": datetime(2024, 3, 1, 9, 30, tzinfo=timezone.utc),
         "booked_at": datetime(2024, 3, 1, 9, 30, tzinfo=timezone.utc)},
        {"region": "east", "policy_count": 12, "claims": None, "premium": 2.0, "active": False, "lapsed": None,
         "written_at": datetime(2024, 3, 2, 10, 0, 0, 250000), "paid_at": None, "booked_at": None},
    ]


def test_the_recorded_shape_reads_as_its_table():
    table = table_from_payload(REGION_COUNTS["columns"], REGION_COUNTS["rows"])
    assert table.schema.types == [pa.string(), pa.int64()]
    assert table.to_pylist() == [{"region": "west", "policy_count": 16666}]


def test_mixed_object_column_becomes_text():
    """A column Superset types `object` holding numbers and strings is text, never a failed read."""
    table = read_execute_response(_answer([{"code": 7}, {"code": "A7"}, {"code": None}, {"code": 2.5}],
                                          [{"name": "code", "type": "object"}]))
    assert table.schema.field("code").type == pa.string()
    assert table.column("code").to_pylist() == ["7", "A7", None, "2.5"]

    # An `object` column of one kind keeps the type pyarrow reads from it: Superset reports an integer column
    # holding a null as `object`.
    numbers = read_execute_response(_answer([{"n": 1}, {"n": None}], [{"name": "n", "type": "object"}]))
    assert numbers.schema.field("n").type == pa.int64()
    assert numbers.column("n").to_pylist() == [1, None]


def test_zero_rows_is_an_empty_table_not_an_error():
    columns = [{"name": "region", "type": "object"}, {"name": "policy_count", "type": "int64"},
               {"name": "written_at", "type": "datetime64[ns]"}]
    table = read_execute_response(_answer([], columns))
    assert table.num_rows == 0
    assert table.column_names == ["region", "policy_count", "written_at"]
    assert table.schema.field("policy_count").type == pa.int64()
    assert table.schema.field("written_at").type == pa.timestamp("us")


def test_cut_response_raises_result_cut():
    cut = _answer(REGION_COUNTS["rows"], REGION_COUNTS["columns"], _response_truncated=True,
                  _truncation_notes=["Result truncated: 1 of 90000 rows returned (limit ~1,000,000 bytes)."])
    with pytest.raises(ResultCut) as caught:
        read_execute_response(cut)
    assert str(caught.value) == ("Superset cut this result to fit its response size; aggregate, or select fewer rows "
                                 "or columns.")


def test_duplicate_column_names_raise():
    """Rows arrive keyed by column name, so two columns of one name cannot both be read from them."""
    doubled = _answer([{"id": 1}], [{"name": "id", "type": "int64"}, {"name": "id", "type": "int64"}])
    with pytest.raises(SupersetQueryFailed) as caught:
        read_execute_response(doubled)
    assert str(caught.value) == "two output columns share the name id; give each its own alias"


def test_a_success_with_no_result_set_is_not_an_empty_answer():
    """Superset answers a statement that returns no rows at all (no SELECT) with neither rows nor columns."""
    with pytest.raises(SupersetQueryFailed) as caught:
        read_execute_response({**_answer([], []), "rows": None, "columns": None, "affected_rows": 0})
    assert str(caught.value) == "the statement returned no result set; send one SELECT"


@pytest.mark.parametrize(("error_type", "raised"), [
    ("timed_out", QueryTimedOut),
    ("stopped", QueryCancelled),
    ("TABLE_SECURITY_ACCESS_ERROR", SupersetAccessRefused),
    ("DATASOURCE_SECURITY_ACCESS_ERROR", SupersetAccessRefused),
    ("DATABASE_SECURITY_ACCESS_ERROR", SupersetAccessRefused),
    ("QUERY_SECURITY_ACCESS_ERROR", SupersetAccessRefused),
    ("DATABASE_NOT_FOUND_ERROR", SupersetAccessRefused),
    ("OAUTH2_REDIRECT", SupersetUnavailable),
    ("OAUTH2_REDIRECT_ERROR", SupersetUnavailable),
    ("failed", SupersetQueryFailed),
    ("INVALID_SQL_ERROR", SupersetQueryFailed),
    ("DML_NOT_ALLOWED_ERROR", SupersetQueryFailed),
    (None, SupersetQueryFailed),
])
def test_each_error_type_raises_its_reading(error_type, raised):
    failed = {"success": False, "rows": None, "columns": None, "error": SUPERSET_TEXT, "error_type": error_type}
    with pytest.raises(raised) as caught:
        read_execute_response(failed)
    assert type(caught.value) is raised
    if raised is SupersetQueryFailed:
        # The statement's own fault, in the words the model repairs from.
        assert str(caught.value) == SUPERSET_TEXT
    else:
        assert SUPERSET_TEXT not in str(caught.value)


def test_a_recorded_live_payload_reads_with_its_types():
    """One payload Superset 7.0.0rc2 sent, recorded by the live suite (`tests/superset_live/test_contract.py`,
    `test_value_encodings`, which also checks it is still what Superset sends): a `numeric`, a `date`, a `timestamp`, a
    `timestamptz`, a `json` value and NULLs, the values synthetic. Superset reports the timestamp `datetime64[us]` and
    the zoned one `datetime64[ns, UTC]`, each value ISO 8601 text; the rest `object`, read as pyarrow infers them:
    the numeric as a number, the date as text, the json value as a struct, and an all-NULL column as null."""
    recorded = json.loads((Path(__file__).parent / "fixtures" / "superset_execute_sql_types.json").read_text())
    table = read_execute_response(recorded)
    assert dict(zip(table.column_names, table.schema.types, strict=True)) == {
        "id": pa.int64(), "premium": pa.float64(), "issued": pa.string(), "issued_at": pa.timestamp("us"),
        "issued_utc": pa.timestamp("us", tz="UTC"),
        "doc": pa.struct([("channel", pa.string()), ("riders", pa.int64())]), "nothing": pa.null()}
    assert table.to_pylist() == [
        {"id": 24, "premium": 988.12, "issued": "2024-06-17", "issued_at": datetime(2024, 6, 17, 9, 30),
         "issued_utc": datetime(2024, 6, 17, 9, 30, tzinfo=timezone.utc), "doc": {"channel": "web", "riders": 0},
         "nothing": None},
        {"id": 25, "premium": 125.25, "issued": "2024-06-24", "issued_at": datetime(2024, 6, 24, 9, 30),
         "issued_utc": datetime(2024, 6, 24, 9, 30, tzinfo=timezone.utc), "doc": None, "nothing": None},
    ]
