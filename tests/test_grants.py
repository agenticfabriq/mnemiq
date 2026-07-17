import json

from mnemiq.authz.grants import DenyAll, FileAuthzProvider, GrantSet
from mnemiq.contract import IdentityContext


def _identity(**kw) -> IdentityContext:
    return IdentityContext(tenant_id="t1", principal_id="u1", **kw)


def _policy(tmp_path, mapping) -> str:
    path = tmp_path / "authz.json"
    path.write_text(json.dumps({"roles": mapping}))
    return str(path)


def test_grants_are_the_union_of_roles_and_groups(tmp_path):
    provider = FileAuthzProvider(
        _policy(tmp_path, {"analyst": ["claim", "policy"], "hr": ["person"]})
    )
    grants = provider.grants_for(_identity(roles=["analyst"], groups=["hr"]))
    assert grants.objects == {"claim", "policy", "person"}
    assert grants.allows("claim") and grants.allows("person")


def test_an_unlisted_object_is_denied(tmp_path):
    provider = FileAuthzProvider(_policy(tmp_path, {"analyst": ["claim"]}))
    grants = provider.grants_for(_identity(roles=["analyst"]))
    assert not grants.allows("person")  # absence of permission is not permission


def test_an_identity_with_no_roles_gets_nothing(tmp_path):
    provider = FileAuthzProvider(_policy(tmp_path, {"analyst": ["claim"]}))
    assert provider.grants_for(_identity()).objects == frozenset()


def test_an_unknown_role_grants_nothing(tmp_path):
    provider = FileAuthzProvider(_policy(tmp_path, {"analyst": ["claim"]}))
    assert provider.grants_for(_identity(roles=["wizard"])).objects == frozenset()


def test_a_missing_policy_file_denies_everything(tmp_path):
    provider = FileAuthzProvider(str(tmp_path / "nope.json"))
    assert provider.grants_for(_identity(roles=["analyst"])).objects == frozenset()


def test_a_corrupt_policy_file_denies_everything(tmp_path):
    path = tmp_path / "authz.json"
    path.write_text("{ not json")
    provider = FileAuthzProvider(str(path))
    assert provider.grants_for(_identity(roles=["analyst"])).objects == frozenset()


def test_deny_all_denies():
    assert not DenyAll().grants_for(_identity(roles=["analyst"])).allows("claim")


def test_fingerprint_is_stable_and_grant_sensitive():
    a = GrantSet(frozenset({"claim", "policy"}))
    b = GrantSet(frozenset({"policy", "claim"}))  # same grants, different order
    c = GrantSet(frozenset({"claim"}))

    assert a.fingerprint == b.fingerprint  # identical access -> shared cache entries
    assert a.fingerprint != c.fingerprint  # broader access must never serve narrower


def test_grantset_writable_defaults_empty_and_allows_write():
    from mnemiq.authz.grants import GrantSet

    g = GrantSet(frozenset({"claim"}))
    assert g.writable == frozenset() and not g.allows_write("claim")
    gw = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}))
    assert gw.allows_write("claim") and not gw.allows_write("policy")
    assert gw.fingerprint == g.fingerprint  # write grant leaves the read-set cache key unchanged


def test_file_authz_dict_form_grants_writes_and_write_implies_read(tmp_path):
    import json

    from mnemiq.authz.grants import FileAuthzProvider
    from mnemiq.contract import IdentityContext

    policy = tmp_path / "authz.json"
    policy.write_text(json.dumps({"roles": {
        "analyst": ["claim"],
        "writer": {"read": ["policy"], "write": ["claim"]},
    }}))
    prov = FileAuthzProvider(str(policy))

    reader = prov.grants_for(IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"]))
    assert reader.objects == frozenset({"claim"}) and reader.writable == frozenset()

    writer = prov.grants_for(IdentityContext(tenant_id="t", principal_id="u", roles=["writer"]))
    assert writer.writable == frozenset({"claim"})
    assert writer.objects == frozenset({"policy", "claim"})  # write implies read


def test_file_authz_malformed_denies_both(tmp_path):
    from mnemiq.authz.grants import FileAuthzProvider
    from mnemiq.contract import IdentityContext

    p = tmp_path / "bad.json"
    p.write_text("not json")
    g = FileAuthzProvider(str(p)).grants_for(IdentityContext(tenant_id="t", principal_id="u"))
    assert g.objects == frozenset() and g.writable == frozenset()


def test_fingerprint_reflects_the_full_policy():
    from mnemiq.authz.grants import GrantSet

    base = GrantSet(frozenset({"claim"}))
    filtered = GrantSet(frozenset({"claim"}), row_filters={"claim": "region = 'US'"})
    cleared = GrantSet(frozenset({"claim"}), pii_clearance=frozenset({"pii"}))
    masked = GrantSet(frozenset({"claim"}), pii_mask=frozenset({"pii"}))
    fps = {base.fingerprint, filtered.fingerprint, cleared.fingerprint, masked.fingerprint}
    assert len(fps) == 4  # every policy dimension changes the authorization boundary
    assert filtered.fingerprint == GrantSet(frozenset({"claim"}),
                                            row_filters={"claim": "region = 'US'"}).fingerprint
    assert hash(filtered) == hash(GrantSet(frozenset({"claim"})))  # dict excluded from hash


def test_file_authz_dict_form_parses_rls_cls(tmp_path):
    import json

    from mnemiq.authz.grants import FileAuthzProvider
    from mnemiq.contract import IdentityContext

    policy = tmp_path / "authz.json"
    policy.write_text(json.dumps({"roles": {
        "us": {"read": ["claim"], "row_filters": {"claim": "region = 'US'"},
               "pii_clearance": ["pii"], "pii_mask": ["phi"]},
        "eu": {"read": ["claim"], "row_filters": {"claim": "region = 'EU'"}},
    }}))
    prov = FileAuthzProvider(str(policy))

    g = prov.grants_for(IdentityContext(tenant_id="t", principal_id="u", roles=["us"]))
    assert g.row_filters == {"claim": "region = 'US'"}
    assert g.pii_clearance == frozenset({"pii"}) and g.pii_mask == frozenset({"phi"})

    both = prov.grants_for(IdentityContext(tenant_id="t", principal_id="u", roles=["us", "eu"]))
    assert both.row_filters["claim"] == "(region = 'US') OR (region = 'EU')"  # OR-combined
