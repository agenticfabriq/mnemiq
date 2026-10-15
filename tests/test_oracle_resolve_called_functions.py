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

    def __init__(self, own=(), synonyms=(), oracle_owned=("SYS",)):
        self.own = set(own)
        # (owner, name, target_owner, target_name, db_link), all as Oracle stores them
        self.synonyms = list(synonyms)
        self.oracle_owned = set(oracle_owned)

    def rows(self, sql, **binds):
        values = set(binds.values())
        if "all_procedures" in sql:
            return [(n,) for n in self.own if n in values]
        if "object_type = 'SYNONYM'" in sql:
            return [(o, n) for o, n, *_ in self.synonyms
                    if o in (binds["owner"], "PUBLIC") and n.lower() in values]
        if "db_link IS NOT NULL" in sql:
            return [(o, n) for o, n, _, _, link in self.synonyms if link]
        if "(table_owner, table_name) IN" in sql:
            pairs = {(binds[f"o{i}"], binds[f"n{i}"]) for i in range(len(binds) // 2)}
            return [(o, n) for o, n, to, tn, _ in self.synonyms if (to, tn) in pairs]
        if "table_name IN" in sql:
            return [(o, n) for o, n, _, tn, _ in self.synonyms if tn in values]
        if "oracle_maintained" in sql:
            return [(u,) for u in self.oracle_owned]
        raise AssertionError(f"unrouted dictionary query: {sql}")


def _adapter(dictionary, resolutions):
    """`resolutions` maps a stored synonym name to what `_name_resolve` returns; a name absent
    from it must never be resolved, and resolving it fails the test."""
    a = OracleAdapter.__new__(OracleAdapter)
    a._schema = "APP"
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
