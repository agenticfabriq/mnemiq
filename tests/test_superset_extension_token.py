"""The Superset extension's token route: the pure minting code and the handler behind the Flask route. Neither
module imports Flask or Superset, so these run in `unit` with no install."""

import os
import time
from types import SimpleNamespace

import jwt
import pytest

from mnemiq.config import Settings
from tests.auth_keys import AUDIENCE, ISSUER, jwks_for, rsa_key
from tests.superset_extension_path import use_extension_source

use_extension_source()

ASK = {"db.read", "superset:sqllab:write", "superset:dataset:read"}  # sub-project 1's set
BUILD = ASK | {"superset:chart:read", "superset:chart:write", "superset:dashboard:read",
               "superset:dashboard:write", "superset:dataset:write"}


def _pem(tmp_path, mode=0o600, name="mint.pem"):
    from cryptography.hazmat.primitives import serialization

    path = tmp_path / name
    path.write_bytes(rsa_key().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                             serialization.NoEncryption()))
    os.chmod(path, mode)
    return path


def _mapping(tmp_path, **over):
    base = {"MNEMIQ_TOKEN_KEY_FILE": str(_pem(tmp_path)), "MNEMIQ_TOKEN_KID": "k1", "MNEMIQ_TOKEN_ISSUER": ISSUER,
            "MNEMIQ_TOKEN_AUDIENCE": AUDIENCE, "MNEMIQ_TOKEN_TENANT": "acme"}
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


def _cfg(tmp_path, **over):
    from agenticfabriq.mnemiq.token import SigningConfig

    return SigningConfig.from_mapping(_mapping(tmp_path, **over))


def _decode(token):
    return jwt.decode(token, rsa_key().public_key(), algorithms=["RS256"], audience=AUDIENCE, issuer=ISSUER)


def test_mint_claims(tmp_path):
    from agenticfabriq.mnemiq.token import mint

    now = 1_800_000_000.0
    token = mint(_cfg(tmp_path), username="ana", purpose="ask", now=now)
    assert jwt.get_unverified_header(token) == {"alg": "RS256", "typ": "JWT", "kid": "k1"}
    found = jwt.decode(token, rsa_key().public_key(), algorithms=["RS256"], audience=AUDIENCE, issuer=ISSUER,
                       options={"verify_exp": False, "verify_nbf": False, "verify_iat": False})
    assert found["iss"] == ISSUER and found["aud"] == AUDIENCE
    assert found["sub"] == "ana" and found["preferred_username"] == "ana"
    assert found["iat"] == found["nbf"] == int(now) and found["exp"] == int(now) + 300
    assert found["tenant"] == "acme" and found["jti"]
    again = mint(_cfg(tmp_path), username="ana", purpose="ask", now=now)
    assert jwt.decode(again, options={"verify_signature": False})["jti"] != found["jti"]


def test_lifetime_is_capped_at_300(tmp_path):
    from agenticfabriq.mnemiq.token import mint

    cfg = _cfg(tmp_path, MNEMIQ_TOKEN_LIFETIME_S="9999")
    assert cfg.lifetime_s == 300
    found = jwt.decode(mint(cfg, username="ana", purpose="ask", now=time.time()), options={"verify_signature": False})
    assert found["exp"] - found["iat"] == 300
    assert _cfg(tmp_path, MNEMIQ_TOKEN_LIFETIME_S="60").lifetime_s == 60
    assert _cfg(tmp_path).lifetime_s == 300
    with pytest.raises(ValueError):
        _cfg(tmp_path, MNEMIQ_TOKEN_LIFETIME_S="0")


def test_purpose_scopes(tmp_path):
    from agenticfabriq.mnemiq.token import mint

    cfg = _cfg(tmp_path)
    now = time.time()
    ask = _decode(mint(cfg, username="ana", purpose="ask", now=now))["scope"]
    build = _decode(mint(cfg, username="ana", purpose="build", now=now))["scope"]
    assert len(ask.split()) == len(ASK) and set(ask.split()) == ASK
    assert set(build.split()) == BUILD
    with pytest.raises(ValueError):
        mint(cfg, username="ana", purpose="admin", now=now)  # type: ignore[arg-type]


def test_refuses_a_world_readable_key(tmp_path):
    from agenticfabriq.mnemiq.token import SigningConfig

    for mode in (0o644, 0o640, 0o604):
        loose = _pem(tmp_path, mode, name=f"loose-{mode:o}.pem")
        with pytest.raises(ValueError, match="0600"):
            SigningConfig.from_mapping(_mapping(tmp_path, MNEMIQ_TOKEN_KEY_FILE=str(loose)))
    link = tmp_path / "link.pem"
    link.symlink_to(_pem(tmp_path))
    with pytest.raises(ValueError, match="regular file"):
        SigningConfig.from_mapping(_mapping(tmp_path, MNEMIQ_TOKEN_KEY_FILE=str(link)))
    with pytest.raises(ValueError, match="regular file"):
        SigningConfig.from_mapping(_mapping(tmp_path, MNEMIQ_TOKEN_KEY_FILE=str(tmp_path)))
    _cfg(tmp_path)  # the control: a 0600 file is accepted


@pytest.mark.parametrize("missing", ["MNEMIQ_TOKEN_KEY_FILE", "MNEMIQ_TOKEN_KID", "MNEMIQ_TOKEN_ISSUER",
                                     "MNEMIQ_TOKEN_AUDIENCE", "MNEMIQ_TOKEN_TENANT"])
def test_refuses_missing_settings(tmp_path, missing):
    from agenticfabriq.mnemiq.token import SigningConfig

    with pytest.raises(ValueError, match=missing):
        SigningConfig.from_mapping(_mapping(tmp_path, **{missing: None}))
    with pytest.raises(ValueError, match=missing):
        SigningConfig.from_mapping(_mapping(tmp_path, **{missing: "  "}))


def test_token_verifies_with_mnemiqs_door(tmp_path):
    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.doors import DoorRefusal, HttpDoors

    from agenticfabriq.mnemiq.token import mint

    cfg = auth_config_from_settings(Settings(tenant="acme", oidc_issuer=ISSUER, oidc_audience=AUDIENCE))
    doors = HttpDoors(cfg, fetch=lambda: jwks_for((rsa_key(), "k1")))
    signing = _cfg(tmp_path)

    def headers(token):
        return {"authorization": [f"Bearer {token}"]}

    fresh = mint(signing, username="ana", purpose="build", now=time.time())
    delegation = doors.v1_delegation(headers(fresh))
    assert delegation.identity.tenant_id == "acme"
    assert delegation.token  # the door hands on the very Bearer it verified
    with pytest.raises(DoorRefusal) as exc:
        doors.v1(headers(mint(signing, username="ana", purpose="build", now=time.time() - 1000)))
    assert exc.value.status == 401


def test_api_ignores_a_username_in_the_body(tmp_path):
    from agenticfabriq.mnemiq.token import BadRequest, mint_for

    cfg = _cfg(tmp_path)
    user = SimpleNamespace(username="ana")
    answer = mint_for(user, {"purpose": "ask", "username": "root", "sub": "root", "preferred_username": "root"},
                      cfg, time.time())
    assert set(answer) == {"token", "expires_in"} and answer["expires_in"] == 300
    found = _decode(answer["token"])
    assert found["sub"] == found["preferred_username"] == "ana"
    for bad in ({}, {"purpose": "x"}, {"purpose": ["ask"]}, None, []):
        with pytest.raises(BadRequest):
            mint_for(user, bad, cfg, time.time())
