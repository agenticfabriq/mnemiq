"""M101: `OracleAdapter.resolve_called_functions` -- the narrower question, rule by rule.

The dictionary and Oracle's resolver are stubbed so each rule is pinned without a database; the
live suite (`test_oracle_adapter.py`, which decides through this path) and the parity run recorded
in M101 are what tie the stubs to Oracle. The rules:

* own functions and package members are defined;
* a synonym here is defined when Oracle resolves it to PL/SQL, whoever owns that;
* a PUBLIC synonym only when it resolves outside Oracle's own schemas;
* a synonym reaching a DB link is unresolvable and is NEVER handed to the resolver, which follows
  a link to the remote -- 40 s on a dead one, then "does not exist";
* a synonym that does not resolve (ORA-06564, 00980, 01775) is unresolvable, never absent;
* one resolving to a table or a view (ORA-04047) is neither.
"""

from __future__ import annotations

import pytest

from mnemiq.adapters import oracle as module
from mnemiq.adapters.oracle import OracleAdapter


class _Dictionary:
    """`_rows` answered from Python sets, routed by the shape of the SQL it is handed."""

    def __init__(self, own=(), synonyms=(), oracle_owned=("SYS",), hidden=(), objects=None):
        self.own = set(own)
        # (owner, name, target_owner, target_name, db_link), all as Oracle stores them
        self.synonyms = list(synonyms)
        self.oracle_owned = set(oracle_owned)
        # What the caller's dictionary does not show, by (owner, name): another schema's private
        # synonym, an object it holds no grant on, a name nothing holds. Every other synonym, and
        # every synonym's target, is shown.
        self.hidden = set(hidden)
        # Shown objects other than synonyms, by (owner, name), with their type: every synonym's
        # target is a FUNCTION unless named here.
        self.objects = dict(objects or {})

    def _shown(self):
        return [s for s in self.synonyms if (s[0], s[1]) not in self.hidden]

    def rows(self, sql, **binds):
        values = set(binds.values())
        if "synonym_name = :name" in sql:
            key = (binds["owner"], binds["name"])
            return [(to, tn, link) for o, n, to, tn, link in self._shown() if (o, n) == key]
        if "object_name = :name" in sql:
            key = (binds["owner"], binds["name"])
            kinds = {(to, tn): "FUNCTION" for _, _, to, tn, _ in self.synonyms}
            kinds.update({(o, n): "SYNONYM" for o, n, *_ in self.synonyms})
            kinds.update(self.objects)
            kind = kinds.get(key) if key not in self.hidden else None
            return [(1,)] if kind is not None and f"'{kind}'" in sql else []
        if "all_procedures" in sql:
            return [(n,) for n in self.own if n in values]
        if "object_type = 'SYNONYM'" in sql:
            return [(o, n) for o, n, *_ in self._shown()
                    if o in (binds["owner"], "PUBLIC") and n.lower() in values]
        if "db_link IS NOT NULL" in sql:
            return [(o, n) for o, n, _, _, link in self._shown() if link]
        if "(table_owner, table_name) IN" in sql:
            pairs = {(binds[f"o{i}"], binds[f"n{i}"]) for i in range(len(binds) // 2)}
            return [(o, n) for o, n, to, tn, _ in self._shown() if (to, tn) in pairs]
        if "table_name IN" in sql:
            return [(o, n) for o, n, _, tn, _ in self._shown() if tn in values]
        if "oracle_maintained" in sql:
            return [(u,) for u in self.oracle_owned]
        raise AssertionError(f"unrouted dictionary query: {sql}")


def _adapter(dictionary, resolutions, schema="APP"):
    """`resolutions` maps a stored synonym name to what `_name_resolve` returns; a name absent
    from it must never be resolved, and resolving it fails the test. The session user is APP;
    `schema` is CURRENT_SCHEMA."""
    a = OracleAdapter.__new__(OracleAdapter)
    a._schema, a._user = schema, "APP"
    a._rows = dictionary.rows

    def resolve(stored):
        if stored not in resolutions:
            raise AssertionError(f"{stored} was handed to the resolver")
        return resolutions[stored]

    a._name_resolve = resolve
    return a


def test_own_functions_and_package_members_are_defined():
    a = _adapter(_Dictionary(own={"app_fn", "pkg_fn"}), {})
    assert a.resolve_called_functions(["app_fn", "pkg_fn", "upper"]) == (["app_fn", "pkg_fn"], [])


@pytest.mark.parametrize("owner,resolves_to,defined", [
    ("APP", "SYS", True),        # this schema's own alias counts, whoever owns its function
    ("PUBLIC", "SYS", False),    # a PUBLIC alias to Oracle's own package does not
    ("PUBLIC", "APP2", True),    # a PUBLIC alias ending outside Oracle's schemas does
])
def test_an_alias_counts_by_who_owns_it_and_where_oracle_resolves_it(owner, resolves_to, defined):
    d = _Dictionary(synonyms=[(owner, "ALIAS", resolves_to, "F", None)])
    a = _adapter(d, {"ALIAS": (resolves_to, None)})
    assert a.resolve_called_functions(["alias"]) == ((["alias"] if defined else []), [])


def test_a_synonym_reaching_a_link_is_unresolvable_and_never_resolved():
    """Directly, through a chain, and through a PUBLIC link the target falls back to."""
    d = _Dictionary(synonyms=[
        ("APP", "VIA_LINK", "APP", "F", "ERP"),           # the link synonym itself
        ("APP", "CHAIN", "APP", "VIA_LINK", None),        # a chain ending at it
        ("PUBLIC", "PUB_LINK", None, "G", "ERP"),         # a PUBLIC link synonym
        ("APP", "FALLBACK", "APP", "PUB_LINK", None),     # APP holds no PUB_LINK: Oracle takes PUBLIC's
    ])
    a = _adapter(d, {})  # resolving any of them fails the test
    assert a.resolve_called_functions(["via_link", "chain", "pub_link", "fallback"]) == (
        [], ["chain", "fallback", "pub_link", "via_link"])


def test_a_cycle_through_a_link_terminates():
    d = _Dictionary(synonyms=[("APP", "A", "APP", "B", None), ("APP", "B", "APP", "A", None),
                              ("APP", "L", None, "X", "ERP"), ("APP", "C", "APP", "L", None),
                              ("APP", "L2", "APP", "C", None)])
    assert _adapter(d, {"A": module._UNRESOLVED, "B": module._UNRESOLVED}).resolve_called_functions(
        ["a", "b", "l2"]) == ([], ["a", "b", "l2"])


@pytest.mark.parametrize("outcome,expected", [
    (module._UNRESOLVED, ([], ["s"])),   # exists, does not resolve: never read as absent
    (module._NOT_PLSQL, ([], [])),       # resolves to a table or a view: no function there
    (("APP2", "RACED"), ([], ["s"])),    # a link that appeared after the link scan
])
def test_what_the_resolver_says_decides_a_synonym(outcome, expected):
    d = _Dictionary(synonyms=[("APP", "S", "APP2", "T", None)])
    assert _adapter(d, {"S": outcome}).resolve_called_functions(["s"]) == expected


def test_a_name_both_defined_and_over_a_link_is_defined():
    """`names` wins an overlap, as in the whole-source walk: the deployer owns a real function."""
    d = _Dictionary(own={"orders"}, synonyms=[("PUBLIC", "ORDERS", None, "ORDERS", "ERP")])
    assert _adapter(d, {}).resolve_called_functions(["orders"]) == (["orders"], [])


def test_a_mixed_case_synonym_is_found_by_its_lowercased_call():
    d = _Dictionary(synonyms=[("APP", "Mixed_Alias", "APP", "F", None)])
    assert _adapter(d, {"Mixed_Alias": ("APP", None)}).resolve_called_functions(
        ["mixed_alias"]) == (["mixed_alias"], [])


@pytest.mark.parametrize("code,expected", [
    (6564, "unresolved"), (980, "unresolved"), (1775, "unresolved"), (4047, "not plsql")])
def test_the_resolver_s_errors_map_to_their_meaning(code, expected):
    """Measured on the fixture: ORA-00980 for a dangling synonym and ORA-01775 for a cycle, each a
    synonym that does not resolve; ORA-04047 for one over a table."""

    class _Error(Exception):
        pass

    class _Cursor:
        def var(self, kind):
            return None

        def callproc(self, name, args):
            raise _Error(type("E", (), {"code": code})())

    a = OracleAdapter.__new__(OracleAdapter)
    a._oracledb = type("M", (), {"DatabaseError": _Error})
    a._probe_timeout_ms = 0
    a._with_cursor = lambda work, timeout: work(_Cursor())
    got = a._name_resolve("S")
    assert got is {"unresolved": module._UNRESOLVED, "not plsql": module._NOT_PLSQL}[expected]


def test_any_other_resolver_error_raises():
    class _Error(Exception):
        pass

    class _Cursor:
        def var(self, kind):
            return None

        def callproc(self, name, args):
            raise _Error(type("E", (), {"code": 3113})())  # end-of-file on channel

    a = OracleAdapter.__new__(OracleAdapter)
    a._oracledb = type("M", (), {"DatabaseError": _Error})
    a._probe_timeout_ms = 0
    a._with_cursor = lambda work, timeout: work(_Cursor())
    with pytest.raises(_Error):
        a._name_resolve("S")


def test_a_public_synonym_sharing_a_name_with_a_private_link_synonym_is_never_resolved():
    """Oracle resolves the name to this schema's synonym first, so resolving the PUBLIC one would
    follow the private one across its link (the review gate)."""
    d = _Dictionary(synonyms=[("APP", "X", "APP", "F", "ERP"), ("PUBLIC", "X", "APP2", "F", None)])
    assert _adapter(d, {}).resolve_called_functions(["x"]) == ([], ["x"])


def test_a_hop_the_dictionary_does_not_show_is_unresolvable_and_never_resolved():
    """M177: ALL_SYNONYMS shows the caller no other schema's private synonym, so a PUBLIC alias to
    one over a DB link looked like any alias, and NAME_RESOLVE followed it across the link: 40 s
    on a dead one, measured. The chain is followed first, through what the dictionary shows, and a
    hop it does not show in another schema is unresolvable unasked, a function without a grant too."""
    d = _Dictionary(synonyms=[("PUBLIC", "APP_PUB", "OTHER", "HOP", None),
                              ("OTHER", "HOP", None, "FN", "DEAD"),
                              ("PUBLIC", "TO_FN", "OTHER", "FN", None)],
                    hidden={("OTHER", "HOP"), ("OTHER", "FN")})
    assert _adapter(d, {}).resolve_called_functions(["app_pub", "to_fn"]) == ([], ["app_pub", "to_fn"])


def test_a_name_this_schema_does_not_hold_is_followed_to_the_public_synonym():
    # Oracle answers a name the schema does not hold by the PUBLIC synonym of that name. The
    # session's own schema is all in its dictionary, so a name missing there is missing.
    d = _Dictionary(synonyms=[("APP", "FB", "APP", "FB_TARGET", None),
                              ("PUBLIC", "FB_TARGET", "APP2", "F", None)],
                    hidden={("APP", "FB_TARGET")})
    assert _adapter(d, {"FB": ("APP2", None)}).resolve_called_functions(["fb"]) == (["fb"], [])


def test_a_name_another_schema_does_not_show_is_not_taken_for_missing():
    # One schema over, a name the dictionary does not show may be one it hides, which Oracle
    # resolves before any PUBLIC synonym, so the fallback is not assumed.
    d = _Dictionary(synonyms=[("APP", "FB", "APP2", "FB_TARGET", None),
                              ("PUBLIC", "FB_TARGET", "APP2", "F", None)],
                    hidden={("APP2", "FB_TARGET")})
    assert _adapter(d, {}).resolve_called_functions(["fb"]) == ([], ["fb"])


def test_a_chain_the_dictionary_shows_is_resolved_as_before():
    d = _Dictionary(synonyms=[("PUBLIC", "P", "APP2", "S", None), ("APP2", "S", "APP2", "F", None)])
    assert _adapter(d, {"P": ("APP2", None)}).resolve_called_functions(["p"]) == (["p"], [])


def test_a_public_alias_sharing_a_name_with_a_private_one_out_of_sight_is_never_resolved():
    # NAME_RESOLVE takes the name, and Oracle reads this schema's synonym of it before PUBLIC's, so
    # resolving the PUBLIC candidate would follow the private chain out of sight.
    d = _Dictionary(synonyms=[("APP", "X", "OTHER", "HOP", None), ("OTHER", "HOP", None, "FN", "DEAD"),
                              ("PUBLIC", "X", "APP2", "F", None)],
                    hidden={("OTHER", "HOP")})
    assert _adapter(d, {}).resolve_called_functions(["x"]) == ([], ["x"])


def test_a_hidden_synonym_in_current_schema_shadowing_a_safe_public_one_is_never_resolved():
    # CURRENT_SCHEMA is OTHER, not the session user. OTHER's private X over a dead link is hidden
    # from the caller, PUBLIC's X is safe, and NAME_RESOLVE('X') starts in OTHER and takes the
    # private one (Codex's review of #194). A name not shown there is out of sight.
    d = _Dictionary(synonyms=[("OTHER", "X", None, "FN", "DEAD"), ("PUBLIC", "X", "APP2", "F", None)],
                    hidden={("OTHER", "X")})
    assert _adapter(d, {}, schema="OTHER").resolve_called_functions(["x"]) == ([], ["x"])


def test_a_synonym_current_schema_shows_is_resolved_from_there():
    d = _Dictionary(synonyms=[("OTHER", "X", "APP2", "F", None)])
    assert _adapter(d, {"X": ("APP2", None)}, schema="OTHER").resolve_called_functions(["x"]) == (["x"], [])


def test_an_object_in_another_namespace_does_not_end_the_chain():
    # Oracle resolves a name among tables, views, sequences, procedures, functions, packages,
    # types, operators and synonyms; an index of the same name is in a namespace of its own and
    # shadows nothing. Taken for the end of the chain, an index X let NAME_RESOLVE follow PUBLIC's X
    # through a hidden link (Codex's second review of #194).
    d = _Dictionary(synonyms=[("PUBLIC", "X", "OTHER", "HOP", None), ("OTHER", "HOP", None, "FN", "DEAD")],
                    hidden={("OTHER", "HOP")}, objects={("APP", "X"): "INDEX"})
    assert _adapter(d, {}).resolve_called_functions(["x"]) == ([], ["x"])


def test_an_object_in_the_shared_namespace_ends_the_chain():
    d = _Dictionary(synonyms=[("PUBLIC", "X", "OTHER", "HOP", None), ("OTHER", "HOP", None, "FN", "DEAD")],
                    hidden={("OTHER", "HOP")}, objects={("APP", "X"): "TABLE"})
    # This schema's table X is what Oracle reads X as: NOT_PLSQL, no function behind the name.
    assert _adapter(d, {"X": module._NOT_PLSQL}).resolve_called_functions(["x"]) == ([], [])
