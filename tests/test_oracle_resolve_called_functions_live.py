"""M101, live: the narrow question agrees with the whole-dictionary read, or Oracle says why not.

Builds one of every synonym shape this adapter has met -- private and PUBLIC, chains, a cycle, a
synonym over a table and over Oracle's own package, live and dead DB links and a chain ending at
one, mixed case, a dangling synonym, a private synonym shadowing a PUBLIC one, and the PUBLIC
fallback -- and asks both questions about every name.

Where they differ, Oracle decides, which is the owner's decision (2026-10-14): a name the
whole-dictionary walk refused and the narrow question clears must FAIL when actually called, and
the one such name is pinned so a new divergence cannot pass unread. The names the narrow question
refuses and the walk did not are synonyms that do not resolve at all, refused on the safe side.

Needs the app user (`MNEMIQ_ORACLE_TEST_*`, with CREATE SYNONYM and CREATE DATABASE LINK) and an
admin for the PUBLIC objects (`MNEMIQ_ORACLE_TEST_ADMIN_*`, connected AS SYSDBA), as the bypass
tests in `test_oracle_adapter.py` do.
"""

from __future__ import annotations

import os

import pytest

DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")
USER = os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser")
PASSWORD = os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw")
ADMIN_USER = os.environ.get("MNEMIQ_ORACLE_TEST_ADMIN_USER")
ADMIN_PASSWORD = os.environ.get("MNEMIQ_ORACLE_TEST_ADMIN_PASSWORD")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (DSN and ADMIN_USER and ADMIN_PASSWORD),
                       reason="set MNEMIQ_ORACLE_TEST_DSN and MNEMIQ_ORACLE_TEST_ADMIN_* "
                              "to compare the narrow question with the whole read"),
]

P = "m101p_"  # every object this test makes, so its cleanup can find them all

PRIVATE = [
    f"CREATE TABLE {P}t (id NUMBER)",
    f"CREATE OR REPLACE FUNCTION {P}fn(x NUMBER) RETURN NUMBER IS BEGIN RETURN x; END;",
    f"CREATE OR REPLACE PACKAGE {P}pkg AS FUNCTION {P}member(x NUMBER) RETURN NUMBER; END;",
    f"CREATE OR REPLACE PACKAGE BODY {P}pkg AS FUNCTION {P}member(x NUMBER) RETURN NUMBER IS "
    "BEGIN RETURN x; END; END;",
    f"CREATE OR REPLACE SYNONYM {P}alias FOR {P}fn",
    f"CREATE OR REPLACE SYNONYM {P}chain_b FOR {P}fn",
    f"CREATE OR REPLACE SYNONYM {P}chain_a FOR {P}chain_b",
    f"CREATE OR REPLACE SYNONYM {P}loop_a FOR {P}loop_b",
    f"CREATE OR REPLACE SYNONYM {P}loop_b FOR {P}loop_a",
    f"CREATE OR REPLACE SYNONYM {P}syn_table FOR {P}t",
    f"CREATE OR REPLACE SYNONYM {P}syn_sys FOR sys.dbms_output",
    f'CREATE OR REPLACE SYNONYM "{P.upper()}Mixed" FOR {P}fn',
    f"CREATE OR REPLACE SYNONYM {P}dangling FOR {P}no_such_thing",
    f"CREATE OR REPLACE SYNONYM {P}shadow FOR {P}t",
    f"CREATE OR REPLACE SYNONYM {P}fb FOR {P}fb_target",
    f"CREATE DATABASE LINK {P}dead CONNECT TO nobody IDENTIFIED BY x USING '10.255.255.1:1521/NOPE'",
    f"CREATE OR REPLACE SYNONYM {P}dead_link_fn FOR {P}fn@{P}dead",
    f"CREATE OR REPLACE SYNONYM {P}chain_to_link FOR {P}dead_link_fn",
]


def _public(owner):
    return [
        f"CREATE PUBLIC SYNONYM {P}pub_fn FOR {owner}.{P}fn",
        f"CREATE PUBLIC SYNONYM {P}pub_sys FOR sys.dbms_output",
        f"CREATE PUBLIC SYNONYM {P}shadow FOR {owner}.{P}fn",
        f"CREATE PUBLIC SYNONYM {P}fb_target FOR {owner}.{P}fn",
        f"CREATE PUBLIC DATABASE LINK {P}pub_dead CONNECT TO nobody IDENTIFIED BY x "
        "USING '10.255.255.1:1521/NOPE'",
        f"CREATE PUBLIC SYNONYM {P}pub_link FOR {P}fn@{P}pub_dead",
        f"CREATE PUBLIC SYNONYM {P}pub_chain_to_link FOR {P}pub_link",
    ]


NAMES = [f"{P}{n}" for n in (
    "fn", "member", "pkg", "alias", "chain_a", "chain_b", "loop_a", "loop_b", "syn_table",
    "syn_sys", "mixed", "dangling", "shadow", "fb", "fb_target", "pub_fn", "pub_sys", "pub_link",
    "pub_chain_to_link", "dead_link_fn", "chain_to_link", "nope")] + ["upper", "nvl", "count"]


def _drop_everything(app_cur, admin_cur, oracledb):
    admin_cur.execute("SELECT synonym_name FROM dba_synonyms WHERE owner = 'PUBLIC' "
                      f"AND synonym_name LIKE '{P.upper()}%'")
    for (name,) in admin_cur.fetchall():
        admin_cur.execute(f'DROP PUBLIC SYNONYM "{name}"')
    for drop in (f"DROP PUBLIC DATABASE LINK {P}pub_dead", f"DROP DATABASE LINK {P}dead"):
        try:
            (admin_cur if "PUBLIC" in drop else app_cur).execute(drop)
        except oracledb.DatabaseError:
            pass
    app_cur.execute("SELECT object_name, object_type FROM user_objects "
                    f"WHERE object_name LIKE '{P.upper()}%' OR object_name LIKE '{P.upper()}Mixed'")
    for name, kind in app_cur.fetchall():
        if kind in ("TABLE", "FUNCTION", "PACKAGE", "SYNONYM"):
            try:
                app_cur.execute(f'DROP {kind} "{name}"')
            except oracledb.DatabaseError:
                pass


def test_the_narrow_question_agrees_with_the_whole_read_or_oracle_refuses_the_call():
    oracledb = pytest.importorskip("oracledb")
    from mnemiq.adapters.oracle import OracleAdapter

    app = oracledb.connect(user=USER, password=PASSWORD, dsn=DSN)
    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    app_cur, admin_cur = app.cursor(), admin.cursor()
    app_cur.call_timeout = 10_000  # the calls below never touch a link; a hang must not pass
    _drop_everything(app_cur, admin_cur, oracledb)  # a run killed before its cleanup
    adapter = OracleAdapter(DSN, USER, PASSWORD)
    try:
        for statement in PRIVATE:
            app_cur.execute(statement)
        for statement in _public(USER.upper()):
            admin_cur.execute(statement)

        whole_defined = set(adapter.user_functions()) & set(NAMES)
        whole_unresolvable = set(adapter.unresolvable_aliases()) & set(NAMES)
        defined, unresolvable = adapter.resolve_called_functions(NAMES)
        whole = whole_defined | whole_unresolvable
        narrow = set(defined) | set(unresolvable)

        cleared = whole - narrow
        assert cleared == {f"{P}shadow"}, cleared  # the one divergence, read and pinned
        for name in cleared:  # ...and Oracle agrees the call reaches no function
            with pytest.raises(oracledb.DatabaseError):
                app_cur.execute(f"SELECT {name}(1) FROM dual")
        assert narrow - whole == {f"{P}dangling", f"{P}loop_a", f"{P}loop_b"}
        # Every link alias refuses, and none was handed to the resolver -- which would have
        # waited 40 s on the dead link and then called it absent.
        assert {f"{P}dead_link_fn", f"{P}chain_to_link", f"{P}pub_link",
                f"{P}pub_chain_to_link"} <= set(unresolvable)
    finally:
        adapter.close()
        _drop_everything(app_cur, admin_cur, oracledb)
        app.close()
        admin.close()
