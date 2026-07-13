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
