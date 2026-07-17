from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.verdict import ApprovedWrite, Refusal, RefusalCode

_VISIBLE = {"claim": {"id", "amount"}, "party": {"id"}}


class _OkAdapter:
    def execute(self, sql):
        return []  # EXPLAIN succeeds


def _grants(writable=("claim",)):
    return GrantSet(frozenset(_VISIBLE), writable=frozenset(writable))


def test_authorized_insert_is_approved():
    v = decide_write("INSERT INTO claim (id, amount) VALUES (1, 2.0)", _VISIBLE, _grants(),
                     adapter=_OkAdapter(), dialect="duckdb")
    assert isinstance(v, ApprovedWrite) and v.target == "claim" and "claim" in v.tables


def test_update_and_delete_require_where():
    for sql in ("UPDATE claim SET amount = 0", "DELETE FROM claim"):
        v = decide_write(sql, _VISIBLE, _grants(), adapter=_OkAdapter(), dialect="duckdb")
        assert isinstance(v, Refusal) and v.code == RefusalCode.UNBOUNDED_WRITE
    ok = decide_write("UPDATE claim SET amount = 0 WHERE id = 1", _VISIBLE, _grants(),
                      adapter=_OkAdapter(), dialect="duckdb")
    assert isinstance(ok, ApprovedWrite) and ok.target == "claim"


def test_ddl_select_and_multistatement_are_not_writes():
    for sql in ("DROP TABLE claim", "SELECT * FROM claim",
                "INSERT INTO claim (id) VALUES (1); DROP TABLE claim"):
        v = decide_write(sql, _VISIBLE, _grants(), adapter=_OkAdapter(), dialect="duckdb")
        assert isinstance(v, Refusal) and v.code in {
            RefusalCode.NOT_A_WRITE, RefusalCode.NOT_A_SINGLE_STATEMENT, RefusalCode.PARSE_ERROR,
        }


def test_target_not_writable_is_refused():
    v = decide_write("INSERT INTO claim (id) VALUES (1)", _VISIBLE, _grants(writable=()),
                     adapter=_OkAdapter(), dialect="duckdb")
    assert isinstance(v, Refusal) and v.code == RefusalCode.UNAUTHORIZED_WRITE


def test_unreadable_referenced_table_is_refused():
    v = decide_write("INSERT INTO claim (id) SELECT id FROM secret", _VISIBLE, _grants(),
                     adapter=_OkAdapter(), dialect="duckdb")
    assert isinstance(v, Refusal) and v.code == RefusalCode.UNAUTHORIZED_TABLE


def test_explain_failure_is_surfaced():
    class _BadAdapter:
        def execute(self, sql):
            raise Exception("relation does not exist")

    v = decide_write("INSERT INTO claim (id) VALUES (1)", _VISIBLE, _grants(),
                     adapter=_BadAdapter(), dialect="duckdb")
    assert isinstance(v, Refusal) and v.code == RefusalCode.EXPLAIN_FAILED
