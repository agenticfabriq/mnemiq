import duckdb
import pytest

from mnemiq.semantic.embedding_width import declared_width, refuse_if_mismatched


def test_declared_width_reads_the_declared_array_size(tmp_path):
    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    con.execute("CREATE TABLE semantic_object (object_id TEXT, embedding FLOAT[1536])")
    assert declared_width(con, "semantic_object") == 1536


def test_declared_width_on_a_table_that_has_never_been_built_is_none(tmp_path):
    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    assert declared_width(con, "semantic_object") is None


def test_declared_width_ignores_a_same_named_table_in_an_attached_catalog(tmp_path):
    """information_schema.columns spans every attached catalog and schema. The old query filtered
    it on table_name + column_name alone, with no table_schema/table_catalog, so `fetchone()`
    picked ARBITRARILY between this database's own semantic_object and an attached catalog's
    table of the same name -- exactly the bug this branch removed three other places
    (federated_build's DELETE ordering, definition_index's catalog match, retrieval's read path).
    DESCRIBE resolves the bare name through the same catalog/schema search path a real query
    against the table would use, so an attached catalog's same-named table cannot decide the
    answer.
    """
    main_path = str(tmp_path / "main.duckdb")
    other_path = str(tmp_path / "aaa_other.duckdb")
    con = duckdb.connect(main_path)
    con.execute("CREATE TABLE semantic_object (object_id TEXT, embedding FLOAT[1536])")
    # Named to sort ahead of the primary connection's own catalog (DuckDB renames a file-stem
    # catalog that collides with the reserved name "main" to "<stem>_db", so the primary catalog
    # here is "main_db") -- information_schema.columns lists catalogs in NAME order, not attach
    # order, so this is what made the old query's `fetchone()` return the wrong row deterministically
    # in this test, not by luck of attach sequence.
    con.execute(f"ATTACH '{other_path}' AS aaa_other")
    con.execute("CREATE TABLE aaa_other.semantic_object (object_id TEXT, embedding FLOAT[64])")

    assert declared_width(con, "semantic_object") == 1536


def test_refuse_if_mismatched_raises_naming_both_widths(tmp_path):
    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    con.execute("CREATE TABLE semantic_object (object_id TEXT, embedding FLOAT[1536])")
    with pytest.raises(RuntimeError) as exc_info:
        refuse_if_mismatched(con, "semantic_object", 1024)
    message = str(exc_info.value)
    assert "1536" in message and "1024" in message


def test_refuse_if_mismatched_is_silent_on_a_never_built_table(tmp_path):
    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    refuse_if_mismatched(con, "semantic_object", 1024)  # nothing to compare against -- no raise


def test_refuse_if_mismatched_is_silent_when_the_width_agrees(tmp_path):
    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    con.execute("CREATE TABLE semantic_object (object_id TEXT, embedding FLOAT[1536])")
    refuse_if_mismatched(con, "semantic_object", 1536)  # same width -- no raise
