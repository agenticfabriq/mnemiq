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


OTHER = "m177_other"
M177_DROPS = ["DROP PUBLIC SYNONYM m177_app_priv", "DROP PUBLIC SYNONYM m177_app_pub",
              "DROP PUBLIC SYNONYM m177_shadow", "DROP PUBLIC SYNONYM m177_idx",
              "DROP PUBLIC DATABASE LINK m177_dead",
              f"DROP USER {OTHER} CASCADE"]


def _drop_m177(admin_cur, oracledb):
    for drop in M177_DROPS:
        try:
            admin_cur.execute(drop)
        except oracledb.DatabaseError:
            pass


def test_a_public_alias_over_another_schema_s_link_synonym_is_never_handed_to_the_resolver():
    """M177: the link walk reads ALL_SYNONYMS, which shows the caller no other schema's private
    synonym, so a PUBLIC alias to one that reaches a DB link looked like any other alias and went to
    NAME_RESOLVE: 40 s on a dead link, DPY-4024 at the adapter's 30 s probe timeout, a crash at 8 s.
    The alias's chain is followed first, and a hop the caller's dictionary does not show, in
    another schema, is unresolvable unasked.

    Two accounts: `m177_other` owns a private synonym over a PUBLIC dead link and another over its
    own private dead link, and a PUBLIC synonym names each. The app user is the caller."""
    import secrets
    import time

    oracledb = pytest.importorskip("oracledb")
    from mnemiq.adapters.oracle import OracleAdapter

    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    ac = admin.cursor()
    _drop_m177(ac, oracledb)  # a run killed before its cleanup
    other_pw = "P" + secrets.token_hex(8)
    adapter = OracleAdapter(DSN, USER, PASSWORD)
    try:
        for statement in (f"CREATE USER {OTHER} IDENTIFIED BY {other_pw}",
                          f"GRANT CREATE SESSION, CREATE SYNONYM, CREATE DATABASE LINK TO {OTHER}",
                          "CREATE PUBLIC DATABASE LINK m177_dead CONNECT TO nobody IDENTIFIED BY x "
                          "USING '10.255.255.1:1521/NOPE'"):
            ac.execute(statement)
        other = oracledb.connect(user=OTHER, password=other_pw, dsn=DSN)
        try:
            for statement in ("CREATE DATABASE LINK m177_priv CONNECT TO nobody IDENTIFIED BY x "
                              "USING '10.255.255.1:1521/NOPE'",
                              "CREATE SYNONYM m177_hop_priv FOR m177_fn@m177_priv",
                              "CREATE SYNONYM m177_hop_pub FOR m177_fn@m177_dead"):
                other.cursor().execute(statement)
        finally:
            other.close()
        ac.execute(f"CREATE PUBLIC SYNONYM m177_app_priv FOR {OTHER}.m177_hop_priv")
        ac.execute(f"CREATE PUBLIC SYNONYM m177_app_pub FOR {OTHER}.m177_hop_pub")

        asked = []
        resolve = adapter._name_resolve

        def spy(stored):
            asked.append(stored)
            if stored.upper().startswith("M177_"):  # over the dead link it would wait 40 s
                raise AssertionError(f"{stored} was handed to NAME_RESOLVE")
            return resolve(stored)

        adapter._name_resolve = spy
        started = time.monotonic()
        result = adapter.resolve_called_functions(["m177_app_priv", "m177_app_pub"])
        assert result == ([], ["m177_app_priv", "m177_app_pub"]), result
        assert asked == [] and time.monotonic() - started < 10, (asked, time.monotonic() - started)
    finally:
        adapter.close()
        _drop_m177(ac, oracledb)
        admin.close()



def test_a_hidden_link_synonym_in_current_schema_shadowing_a_safe_public_one_is_never_resolved():
    """M177, with CURRENT_SCHEMA another account's: NAME_RESOLVE starts there, where `m177_other`
    holds a private synonym over a dead link that the caller cannot see, shadowing a PUBLIC synonym
    of the same name over the caller's own function (Codex's review of #194). From the caller's own
    schema, the control, the same name resolves to that function."""
    import secrets

    oracledb = pytest.importorskip("oracledb")
    from mnemiq.adapters.oracle import OracleAdapter

    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    ac = admin.cursor()
    app = oracledb.connect(user=USER, password=PASSWORD, dsn=DSN)
    _drop_m177(ac, oracledb)
    other_pw = "P" + secrets.token_hex(8)
    shadowed = OracleAdapter(DSN, USER, PASSWORD, schema=OTHER)
    own = OracleAdapter(DSN, USER, PASSWORD)
    try:
        for statement in (f"CREATE USER {OTHER} IDENTIFIED BY {other_pw}",
                          f"GRANT CREATE SESSION, CREATE SYNONYM TO {OTHER}",
                          "CREATE PUBLIC DATABASE LINK m177_dead CONNECT TO nobody IDENTIFIED BY x "
                          "USING '10.255.255.1:1521/NOPE'",
                          f"CREATE PUBLIC SYNONYM m177_shadow FOR {USER.upper()}.m177_safe_fn"):
            ac.execute(statement)
        app.cursor().execute("CREATE OR REPLACE FUNCTION m177_safe_fn(x NUMBER) RETURN NUMBER IS "
                             "BEGIN RETURN x; END;")
        other = oracledb.connect(user=OTHER, password=other_pw, dsn=DSN)
        try:
            other.cursor().execute("CREATE SYNONYM m177_shadow FOR m177_fn@m177_dead")
        finally:
            other.close()

        def spy(stored):  # over the dead link it would wait 40 s
            raise AssertionError(f"{stored} was handed to NAME_RESOLVE")

        shadowed._name_resolve = spy
        assert shadowed.resolve_called_functions(["m177_shadow"]) == ([], ["m177_shadow"])
        assert own.resolve_called_functions(["m177_shadow"]) == (["m177_shadow"], [])  # the control
    finally:
        shadowed.close()
        own.close()
        _drop_m177(ac, oracledb)
        try:
            app.cursor().execute("DROP FUNCTION m177_safe_fn")
        except oracledb.DatabaseError:
            pass
        app.close()
        admin.close()



def test_an_index_named_like_a_public_alias_over_a_hidden_link_ends_nothing():
    """M177: the caller's own index shares its name with a PUBLIC alias whose chain runs through
    another account's hidden synonym over a dead link. An index is in a namespace of its own, so
    Oracle resolves the name to the PUBLIC alias; taken for the end of the chain, it let the name
    reach NAME_RESOLVE (Codex's second review of #194)."""
    import secrets

    oracledb = pytest.importorskip("oracledb")
    from mnemiq.adapters.oracle import OracleAdapter

    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    ac = admin.cursor()
    app = oracledb.connect(user=USER, password=PASSWORD, dsn=DSN)
    _drop_m177(ac, oracledb)
    for drop in ("DROP TABLE m177_t",):
        try:
            app.cursor().execute(drop)
        except oracledb.DatabaseError:
            pass
    other_pw = "P" + secrets.token_hex(8)
    adapter = OracleAdapter(DSN, USER, PASSWORD)
    try:
        for statement in (f"CREATE USER {OTHER} IDENTIFIED BY {other_pw}",
                          f"GRANT CREATE SESSION, CREATE SYNONYM TO {OTHER}",
                          "CREATE PUBLIC DATABASE LINK m177_dead CONNECT TO nobody IDENTIFIED BY x "
                          "USING '10.255.255.1:1521/NOPE'",
                          f"CREATE PUBLIC SYNONYM m177_idx FOR {OTHER}.m177_hop_pub"):
            ac.execute(statement)
        for statement in ("CREATE TABLE m177_t (id NUMBER)", "CREATE INDEX m177_idx ON m177_t (id)"):
            app.cursor().execute(statement)
        other = oracledb.connect(user=OTHER, password=other_pw, dsn=DSN)
        try:
            other.cursor().execute("CREATE SYNONYM m177_hop_pub FOR m177_fn@m177_dead")
        finally:
            other.close()

        def spy(stored):  # over the dead link it would wait 40 s
            raise AssertionError(f"{stored} was handed to NAME_RESOLVE")

        adapter._name_resolve = spy
        assert adapter.resolve_called_functions(["m177_idx"]) == ([], ["m177_idx"])
    finally:
        adapter.close()
        _drop_m177(ac, oracledb)
        try:
            app.cursor().execute("DROP TABLE m177_t")
        except oracledb.DatabaseError:
            pass
        app.close()
        admin.close()
