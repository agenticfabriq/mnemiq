"""From a verified token's claims to an IdentityContext: roles from the claim or the map, groups only when mapped or
opted in, and the tenant always checked."""

import json
import logging

import pytest

from mnemiq.auth.config import AuthConfig, OidcSettings, OpenWebUISettings
from tests.auth_keys import AUDIENCE, ISSUER, claims, owui_claims


@pytest.fixture(autouse=True)
def _fresh_skip_log(monkeypatch):
    import mnemiq.auth.resolve as resolve

    monkeypatch.setattr(resolve, "_SKIP_LOGGED", set())


def _cfg(tmp_path=None, rules=None, *, tenant="acme", groups_claim=None, roles_claim="realm_access.roles",
         openwebui=True) -> AuthConfig:
    map_path = None
    if rules is not None:
        path = tmp_path / "identity-map.json"
        path.write_text(json.dumps(rules))
        map_path = str(path)
    oidc = OidcSettings(issuer=ISSUER, audience=AUDIENCE, openwebui_audience=None, jwks_url="https://idp.example/c",
                        jwks_file=None, tenant_claim="tenant", roles_claim=roles_claim, groups_claim=groups_claim,
                        leeway_s=30.0)
    ow = OpenWebUISettings(secret=b"s" * 32, header="x-openwebui-user-jwt", max_lifetime_s=300,
                           leeway_s=30.0) if openwebui else None
    return AuthConfig(tenant=tenant, oidc=oidc, openwebui=ow, binding=None, user_headers=(), map_path=map_path,
                      mcp_http=False)


def _oidc(found, cfg):
    from mnemiq.auth.resolve import identity_from_oidc

    return identity_from_oidc(found, cfg)


def _rejected(found, cfg, reason, kind=None):
    from mnemiq.auth.token import TokenRejected

    with pytest.raises(kind or TokenRejected) as exc:
        _oidc(found, cfg)
    assert reason in exc.value.reason, exc.value.reason
    return exc.value


def test_roles_come_from_the_claim_when_it_is_present():
    identity = _oidc(claims(realm_access={"roles": [" analyst", "viewer"]}), _cfg())
    assert identity.principal_id == "user-1" and identity.tenant_id == "acme"
    assert identity.roles == ["analyst", "viewer"] and identity.groups == []


def test_roles_come_from_the_map_when_the_claim_is_absent(tmp_path):
    cfg = _cfg(tmp_path, {ISSUER: {"sub": {"user-1": ["store1_manager"]}}})
    assert _oidc(claims(realm_access=None), cfg).roles == ["store1_manager"]


def test_a_client_role_path_is_read(tmp_path):
    cfg = _cfg(roles_claim="resource_access.mnemiq.roles")
    found = claims(realm_access=None, resource_access={"mnemiq": {"roles": ["analyst"]}})
    assert _oidc(found, cfg).roles == ["analyst"]


def test_an_empty_roles_claim_skips_sub_and_email_rules(tmp_path):
    cfg = _cfg(tmp_path, {ISSUER: {"sub": {"user-1": ["analyst"]}, "email": {"a@x.example": ["analyst"]}}})
    found = claims(realm_access={"roles": []}, email="a@x.example", email_verified=True)
    assert _oidc(found, cfg).roles == []


def test_an_empty_roles_claim_still_takes_a_mapped_groups_role(tmp_path):
    cfg = _cfg(tmp_path, {ISSUER: {"groups": {"finance": ["analyst"]}}})
    assert _oidc(claims(realm_access={"roles": []}, groups=["finance"]), cfg).roles == ["analyst"]


def test_a_groups_rule_adds_to_a_present_roles_claim(tmp_path):
    cfg = _cfg(tmp_path, {ISSUER: {"groups": {"finance": ["auditor"]}}})
    assert _oidc(claims(groups=["finance"]), cfg).roles == ["analyst", "auditor"]


@pytest.mark.parametrize("value", ["analyst", {"roles": "analyst"}, [1, 2], ["analyst", None]])
def test_a_roles_claim_of_the_wrong_type_is_rejected(value):
    found = claims(realm_access=value) if not isinstance(value, dict) else claims(realm_access=value)
    if isinstance(value, list):
        found = claims(realm_access={"roles": value})
    _rejected(found, _cfg(), "wrong type")


def test_a_roles_path_whose_middle_is_not_an_object_is_the_wrong_type(tmp_path):
    """Review focus: read as absent, it would open the map's sub rules for a token the provider sent roles in."""
    cfg = _cfg(tmp_path, {ISSUER: {"sub": {"user-1": ["analyst"]}}})
    _rejected(claims(realm_access="admin"), cfg, "wrong type")


def test_token_groups_grant_nothing_without_the_opt_in():
    identity = _oidc(claims(realm_access=None, groups=["analyst"]), _cfg())
    assert identity.groups == [] and identity.roles == []


def test_the_opt_in_passes_groups_through():
    assert _oidc(claims(groups=["analyst", " ops"]), _cfg(groups_claim="groups")).groups == ["analyst", "ops"]


def test_the_opt_in_reads_its_own_claim():
    found = claims(groups=["analyst"], teams=["ops"])
    assert _oidc(found, _cfg(groups_claim="teams")).groups == ["ops"]


def test_a_groups_claim_of_the_wrong_type_is_rejected(tmp_path):
    cfg = _cfg(tmp_path, {ISSUER: {"groups": {"finance": ["analyst"]}}})
    _rejected(claims(groups="finance"), cfg, "wrong type")


def test_an_email_rule_needs_email_verified_true(tmp_path):
    cfg = _cfg(tmp_path, {ISSUER: {"email": {"a@x.example": ["analyst"]}}})
    for verified in (None, False, "true"):
        assert _oidc(claims(realm_access=None, email="a@x.example", email_verified=verified), cfg).roles == []
    identity = _oidc(claims(realm_access=None, email="a@x.example", email_verified=True), cfg)
    assert identity.roles == ["analyst"] and identity.email == "a@x.example"


def test_an_unverified_email_never_reaches_the_identity():
    assert _oidc(claims(email="a@x.example"), _cfg()).email is None


def test_a_missing_tenant_claim_is_rejected():
    _rejected(claims(tenant=None), _cfg(), "tenant")


def test_another_tenant_is_a_tenant_mismatch():
    from mnemiq.auth.resolve import TenantMismatch

    rejected = _rejected(claims(tenant="globex"), _cfg(), "tenant", TenantMismatch)
    assert rejected.sub == "user-1"


def test_tenant_star_accepts_any_and_keeps_the_tokens():
    assert _oidc(claims(tenant="globex"), _cfg(tenant="*")).tenant_id == "globex"


def test_a_map_broken_after_a_good_read_denies_the_next_request(tmp_path):
    """No last-good copy: a revoked role cannot outlive a broken edit."""
    from mnemiq.auth.identity_map import IdentityMapUnavailable

    cfg = _cfg(tmp_path, {ISSUER: {"sub": {"user-1": ["analyst"]}}})
    assert _oidc(claims(realm_access=None), cfg).roles == ["analyst"]
    with open(cfg.map_path, "w") as fh:
        fh.write("{broken")
    with pytest.raises(IdentityMapUnavailable):
        _oidc(claims(realm_access=None), cfg)


def test_removing_a_rule_revokes_it_at_once(tmp_path):
    cfg = _cfg(tmp_path, {ISSUER: {"sub": {"user-1": ["analyst"]}}})
    assert _oidc(claims(realm_access=None), cfg).roles == ["analyst"]
    with open(cfg.map_path, "w") as fh:
        json.dump({ISSUER: {}}, fh)
    assert _oidc(claims(realm_access=None), cfg).roles == []


def test_a_present_claim_beside_sub_rules_is_logged_once_per_issuer(tmp_path, caplog):
    cfg = _cfg(tmp_path, {ISSUER: {"sub": {"user-1": ["analyst"]}}})
    with caplog.at_level(logging.INFO, logger="mnemiq.auth"):
        _oidc(claims(), cfg)
        _oidc(claims(sub="user-2"), cfg)
    assert caplog.text.count("skipped") == 1


def test_open_webui_roles_come_from_sub_rules_only(tmp_path):
    from mnemiq.auth.resolve import identity_from_openwebui

    cfg = _cfg(tmp_path, {"open-webui": {"sub": {"owui-user-1": ["store1_manager"]}}})
    identity = identity_from_openwebui(owui_claims(role="admin"), cfg)
    assert identity.roles == ["store1_manager"]
    assert identity.tenant_id == "acme" and identity.email is None and identity.groups == []
    assert identity.principal_id == "open-webui:owui-user-1"


def test_open_webuis_admin_grants_no_role(tmp_path):
    from mnemiq.auth.resolve import identity_from_openwebui

    cfg = _cfg(tmp_path, {"open-webui": {"sub": {}}})
    assert identity_from_openwebui(owui_claims(role="admin"), cfg).roles == []
    assert identity_from_openwebui(owui_claims(role="admin"), _cfg()).roles == []


def test_an_unmapped_group_named_like_a_role_grants_and_offers_nothing(tmp_path):
    """The join, from the token to the policy and the profiles: a group called `analyst` grants `analyst`'s tables
    only through a groups rule, and removing the rule revokes them."""
    from mnemiq.authz.grants import FileAuthzProvider
    from mnemiq.config import Settings
    from mnemiq.profiles import Profile, offered_to

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"roles": {"analyst": ["claim"]}}))
    authz = FileAuthzProvider(str(policy))
    profiles = {"default": Profile("default", None, Settings()),
                "analysts": Profile("analysts", frozenset({"analyst"}), Settings())}
    found = claims(realm_access=None, groups=["analyst"])

    cfg = _cfg(tmp_path, {ISSUER: {}})
    identity = _oidc(found, cfg)
    assert authz.grants_for(identity).objects == frozenset()
    assert offered_to(identity, profiles) == ["default"]

    with open(cfg.map_path, "w") as fh:
        json.dump({ISSUER: {"groups": {"analyst": ["analyst"]}}}, fh)
    identity = _oidc(found, cfg)
    assert authz.grants_for(identity).objects == frozenset({"claim"})
    assert offered_to(identity, profiles) == ["analysts", "default"]

    with open(cfg.map_path, "w") as fh:
        json.dump({ISSUER: {}}, fh)
    assert authz.grants_for(_oidc(found, cfg)).objects == frozenset()
