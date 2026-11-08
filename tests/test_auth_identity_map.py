"""The identity map: roles for subjects, groups and verified emails, keyed by issuer, failing closed and loudly."""

import json
import logging

import pytest

from tests.auth_keys import ISSUER

OWUI = "open-webui"
ISSUERS = {ISSUER, OWUI}


def _write(tmp_path, body) -> str:
    path = tmp_path / "identity-map.json"
    path.write_text(body if isinstance(body, str) else json.dumps(body))
    return str(path)


def _read(path, issuers=ISSUERS, **kw):
    from mnemiq.auth.identity_map import IdentityMap

    return IdentityMap.read(path, issuers=issuers, **kw)


def _startup(path, issuers=ISSUERS, groups_passthrough=False):
    from mnemiq.auth.identity_map import IdentityMap

    IdentityMap.validate_for_startup(path, issuers=issuers, groups_passthrough=groups_passthrough,
                                     oidc_issuer=ISSUER)


def test_an_email_rule_under_open_webui_refuses_to_start(tmp_path):
    from mnemiq.auth.config import AuthConfigError

    with pytest.raises(AuthConfigError, match="open-webui"):
        _startup(_write(tmp_path, {OWUI: {"email": {"ana@example.com": ["analyst"]}}}))


def test_a_groups_rule_under_open_webui_refuses_to_start(tmp_path):
    from mnemiq.auth.config import AuthConfigError

    with pytest.raises(AuthConfigError, match="open-webui"):
        _startup(_write(tmp_path, {OWUI: {"groups": {"finance": ["analyst"]}}}))


def test_a_groups_rule_beside_the_opt_in_refuses_to_start(tmp_path):
    from mnemiq.auth.config import AuthConfigError

    with pytest.raises(AuthConfigError, match="MNEMIQ_OIDC_GROUPS_CLAIM"):
        _startup(_write(tmp_path, {ISSUER: {"groups": {"finance": ["analyst"]}}}), groups_passthrough=True)


def test_a_sub_rule_beside_the_opt_in_starts(tmp_path):
    _startup(_write(tmp_path, {ISSUER: {"sub": {"u1": ["analyst"]}}}), groups_passthrough=True)


@pytest.mark.parametrize("body", [
    "{not json",
    json.dumps([1, 2]),
    json.dumps({ISSUER: {"subject": {"u1": ["analyst"]}}}),  # unknown claim key
    json.dumps({"https://elsewhere.example": {"sub": {"u1": ["analyst"]}}}),  # unknown issuer key
    json.dumps({ISSUER: {"sub": {"u1": "analyst"}}}),  # roles not a list
    json.dumps({ISSUER: {"sub": {"u1": ["analyst", 3]}}}),  # a role not a string
    json.dumps({ISSUER: ["sub"]}),
])
def test_an_invalid_map_denies_every_identity_and_logs(tmp_path, caplog, body):
    from mnemiq.auth.identity_map import IdentityMapUnavailable

    path = _write(tmp_path, body)
    with caplog.at_level(logging.ERROR, logger="mnemiq.auth"), pytest.raises(IdentityMapUnavailable):
        _read(path)
    assert "identity map" in caplog.text


def test_an_unreadable_map_denies_every_identity_and_logs(tmp_path, caplog):
    from mnemiq.auth.identity_map import IdentityMapUnavailable

    with caplog.at_level(logging.ERROR, logger="mnemiq.auth"), pytest.raises(IdentityMapUnavailable):
        _read(str(tmp_path / "missing.json"))
    assert "identity map" in caplog.text


def test_an_invalid_map_at_startup_logs_and_starts(tmp_path, caplog):
    """Startup refuses only the rules that contradict the configuration; a file it cannot read is an outage the
    doors answer with 503 on every request until it is fixed, the access policy's own rule."""
    with caplog.at_level(logging.ERROR, logger="mnemiq.auth"):
        _startup(_write(tmp_path, "{not json"))
    assert "identity map" in caplog.text


def test_the_map_never_logs_its_values(tmp_path, caplog):
    from mnemiq.auth.identity_map import IdentityMapUnavailable

    path = _write(tmp_path, {ISSUER: {"email": {"private.person@example.com": "analyst"}}})
    with caplog.at_level(logging.DEBUG), pytest.raises(IdentityMapUnavailable):
        _read(path)
    assert "private.person" not in caplog.text


def test_the_same_sub_under_two_issuers_maps_only_under_its_own(tmp_path):
    m = _read(_write(tmp_path, {ISSUER: {"sub": {"u1": ["analyst"]}}, OWUI: {"sub": {"u1": ["store1_manager"]}}}))
    assert m.roles_for(ISSUER, sub="u1", email=None, groups=[], roles_claim_present=False) == ["analyst"]
    assert m.roles_for(OWUI, sub="u1", email=None, groups=[], roles_claim_present=False) == ["store1_manager"]


def test_the_same_group_under_two_issuers_maps_only_under_its_own(tmp_path):
    other = "https://second.example/realms/r"
    m = _read(_write(tmp_path, {ISSUER: {"groups": {"finance": ["analyst"]}},
                                other: {"groups": {"finance": ["store1_manager"]}}}), issuers={ISSUER, other})
    assert m.roles_for(ISSUER, sub="x", email=None, groups=["finance"], roles_claim_present=False) == ["analyst"]
    assert m.roles_for(other, sub="x", email=None, groups=["finance"], roles_claim_present=False) == [
        "store1_manager"]


def test_email_matching_folds_ascii_case_on_both_sides(tmp_path):
    m = _read(_write(tmp_path, {ISSUER: {"email": {"Ana@Example.com": ["analyst"], "Straße@x.example": ["viewer"]}}}))
    assert m.roles_for(ISSUER, sub="x", email="ana@example.COM", groups=[], roles_claim_present=False) == ["analyst"]
    assert m.roles_for(ISSUER, sub="x", email="STRAßE@X.example", groups=[], roles_claim_present=False) == [
        "viewer"]


@pytest.mark.parametrize("rule, sent", [
    ("ana@fass.de", "ana@faß.de"),  # two registrable domains under IDNA2008
    ("strasse@x.example", "Straße@x.example"),
    ("kai@x.example", "\u212aai@x.example"),  # the Kelvin sign folds to k under Unicode case folding
])
def test_email_matching_never_merges_distinct_unicode_addresses(tmp_path, rule, sent):
    m = _read(_write(tmp_path, {ISSUER: {"email": {rule: ["analyst"]}}}))
    assert m.roles_for(ISSUER, sub="x", email=sent, groups=[], roles_claim_present=False) == []


def test_sub_and_group_match_exactly(tmp_path):
    m = _read(_write(tmp_path, {ISSUER: {"sub": {"U1": ["analyst"]}, "groups": {"Finance": ["viewer"]}}}))
    assert m.roles_for(ISSUER, sub="u1", email=None, groups=["finance"], roles_claim_present=False) == []


def test_roles_union_across_matching_rules(tmp_path):
    m = _read(_write(tmp_path, {ISSUER: {"sub": {"u1": ["analyst", " viewer"]}, "groups": {"g": ["viewer", "ops"]},
                                         "email": {"a@x.example": ["auditor"]}}}))
    assert m.roles_for(ISSUER, sub="u1", email="a@x.example", groups=["g"], roles_claim_present=False) == [
        "analyst", "auditor", "ops", "viewer"]


def test_a_present_roles_claim_skips_sub_and_email_rules_but_not_groups(tmp_path):
    m = _read(_write(tmp_path, {ISSUER: {"sub": {"u1": ["analyst"]}, "email": {"a@x.example": ["auditor"]},
                                         "groups": {"g": ["viewer"]}}}))
    assert m.roles_for(ISSUER, sub="u1", email="a@x.example", groups=["g"], roles_claim_present=True) == ["viewer"]


def test_a_mapped_role_the_policy_does_not_declare_warns(tmp_path, caplog):
    from mnemiq.authz.grants import FileAuthzProvider

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"roles": {"analyst": ["claim"]}}))
    m = _read(_write(tmp_path, {ISSUER: {"sub": {"u1": ["analyst", "anlyst"]}}}))
    with caplog.at_level(logging.WARNING, logger="mnemiq.auth"):
        m.warn_undeclared_roles(FileAuthzProvider(str(policy)))
    assert "anlyst" in caplog.text and "'analyst'" not in caplog.text


def test_the_startup_check_runs_from_the_settings(tmp_path):
    """The join: a conflicting rule stops the server through the same call the server makes at startup."""
    from mnemiq.auth.config import AuthConfigError, auth_config_from_settings
    from mnemiq.config import Settings

    path = _write(tmp_path, {OWUI: {"email": {"ana@example.com": ["analyst"]}}})
    with pytest.raises(AuthConfigError, match="open-webui"):
        auth_config_from_settings(Settings(openwebui_jwt_secret="s" * 32, openwebui_connection_key_header="X-K",
                                           openwebui_connection_key="k" * 32, tenant="acme",
                                           identity_map_path=path))
