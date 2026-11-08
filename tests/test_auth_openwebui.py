"""Open WebUI's signed user header: HS256 under its own secret, issued by `open-webui`, short-lived."""

import json
import logging
import time

import pytest

from mnemiq.auth.config import OpenWebUISettings
from tests.auth_keys import claims, hs256, jwk, mint, owui_claims, public_pem, raw_token, rsa_key

SECRET = b"s" * 32


def _verifier(**over):
    from mnemiq.auth.openwebui import OpenWebUIVerifier

    base = dict(secret=SECRET, header="x-openwebui-user-jwt", max_lifetime_s=300, leeway_s=30.0)
    base.update(over)
    return OpenWebUIVerifier(OpenWebUISettings(**base))


def _rejected(token, reason: str, **over):
    from mnemiq.auth.token import TokenRejected

    with pytest.raises(TokenRejected) as exc:
        _verifier(**over).claims(token)
    assert reason in exc.value.reason, exc.value.reason
    return exc.value


def test_a_valid_token_yields_its_claims():
    got = _verifier().claims(hs256(owui_claims(), SECRET))
    assert got["sub"] == "owui-user-1" and got["iss"] == "open-webui"


def test_the_wrong_secret_is_rejected():
    _rejected(hs256(owui_claims(), b"t" * 32), "signature")


def test_another_issuer_is_rejected():
    _rejected(hs256(owui_claims(iss="https://idp.example/realms/r"), SECRET), "issuer")


def test_an_expired_token_is_rejected():
    now = int(time.time())
    _rejected(hs256(owui_claims(iat=now - 400, exp=now - 100), SECRET), "expired")


def test_a_lifetime_above_the_maximum_is_rejected():
    now = int(time.time())
    _rejected(hs256(owui_claims(iat=now, exp=now + 301), SECRET), "lifetime")


def test_the_maximum_lifetime_itself_verifies():
    now = int(time.time())
    assert _verifier().claims(hs256(owui_claims(iat=now, exp=now + 300), SECRET))["sub"] == "owui-user-1"


@pytest.mark.parametrize("missing", ["sub", "email", "iat", "exp"])
def test_a_missing_required_claim_is_rejected(missing):
    _rejected(hs256(owui_claims(**{missing: None}), SECRET), missing)


def test_alg_none_is_rejected():
    _rejected(raw_token(owui_claims(), alg="none", kid=None), "algorithm")


def test_an_rs256_token_is_rejected():
    _rejected(mint(owui_claims()), "algorithm")


def test_an_identity_provider_token_signed_with_the_secret_is_rejected():
    """An IdP-shaped token (its issuer, audience and roles) under HS256 with this secret is still not Open WebUI's."""
    _rejected(hs256(claims(), SECRET), "")


def test_an_hmac_keyed_with_the_identity_providers_public_key_is_rejected():
    """M9's "HS256 signed with the public key", for this slot: the JWKS is public, so its bytes are no secret."""
    for key_bytes in (public_pem(rsa_key()), json.dumps(jwk(rsa_key(), "k1")).encode()):
        _rejected(raw_token(owui_claims(), alg="HS256", key=key_bytes, kid=None), "signature")


def test_pending_is_refused():
    _rejected(hs256(owui_claims(role="pending"), SECRET), "pending")


def test_admin_passes_the_verifier():
    """Open WebUI's role is not a data role; what roles an identity gets is the identity map's call."""
    assert _verifier().claims(hs256(owui_claims(role="admin"), SECRET))["role"] == "admin"


def test_the_verifier_is_keyed_only_by_its_own_secret():
    import inspect

    from mnemiq.auth.openwebui import OpenWebUIVerifier

    assert list(inspect.signature(OpenWebUIVerifier.__init__).parameters) == ["self", "ow"]


def test_the_email_is_never_logged(caplog):
    from mnemiq.auth.token import TokenRejected

    v = _verifier()
    with caplog.at_level(logging.DEBUG):
        v.claims(hs256(owui_claims(email="secret-person@example.com"), SECRET))
        with pytest.raises(TokenRejected):
            v.claims(hs256(owui_claims(email="secret-person@example.com", role="pending"), SECRET))
    assert "secret-person" not in caplog.text


def test_a_rejection_names_the_sub_only_after_the_signature_verified():
    assert _rejected(hs256(owui_claims(role="pending"), SECRET), "pending").sub == "owui-user-1"
    assert _rejected(hs256(owui_claims(), b"t" * 32), "signature").sub is None
