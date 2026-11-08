"""The identity provider's token verifier: signature, issuer, audience and lifetime, with keys from a JWKS."""

import logging
import socket
import threading
import time

import anyio
import pytest

from mnemiq.auth.config import OidcSettings
from tests.auth_keys import (AUDIENCE, ISSUER, OWUI_AUDIENCE, claims, ec_key, jwks_for, mint, public_pem,
                             raw_token, rsa_key)


def _oidc(**over) -> OidcSettings:
    base = dict(issuer=ISSUER, audience=AUDIENCE, openwebui_audience=OWUI_AUDIENCE, jwks_url="https://idp.example/certs",
                jwks_file=None, tenant_claim="tenant", roles_claim="realm_access.roles", groups_claim=None,
                leeway_s=30.0)
    base.update(over)
    return OidcSettings(**base)


class _Fetch:
    """A JWKS endpoint the test controls: what it serves, and how often it was asked."""

    def __init__(self, jwks=None, error=None):
        self.jwks, self.error, self.calls = jwks, error, 0

    def __call__(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.jwks


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _verifier(fetch=None, *, audience=AUDIENCE, refuse=OWUI_AUDIENCE, clock=None, **oidc):
    from mnemiq.auth.token import JwtIdentityVerifier

    return JwtIdentityVerifier(_oidc(**oidc), audience=audience, refuse_audience=refuse,
                               fetch=fetch or _Fetch(jwks_for((rsa_key(), "k1"), (ec_key(), "e1"))),
                               clock=clock or _Clock())


def _rejected(verifier, token, reason: str):
    from mnemiq.auth.token import TokenRejected

    with pytest.raises(TokenRejected) as exc:
        verifier.claims(token)
    assert reason in exc.value.reason, exc.value.reason
    return exc.value


def test_a_valid_token_yields_its_claims():
    got = _verifier().claims(mint(claims()))
    assert got["sub"] == "user-1" and got["tenant"] == "acme"


def test_es256_verifies():
    assert _verifier().claims(mint(claims(), ec_key(), alg="ES256", kid="e1"))["sub"] == "user-1"


def test_a_wrong_audience_is_rejected():
    _rejected(_verifier(), mint(claims(aud="another-service")), "audience")


def test_a_wrong_issuer_is_rejected():
    _rejected(_verifier(), mint(claims(iss="https://elsewhere.example/realms/r")), "issuer")


def test_expired_past_the_leeway_is_rejected():
    now = int(time.time())
    _rejected(_verifier(), mint(claims(exp=now - 31, iat=now - 400, nbf=now - 400)), "expired")


def test_expired_within_the_leeway_still_verifies():
    now = int(time.time())
    assert _verifier().claims(mint(claims(exp=now - 29, iat=now - 400, nbf=now - 400)))["sub"] == "user-1"


def test_not_yet_valid_past_the_leeway_is_rejected():
    now = int(time.time())
    _rejected(_verifier(), mint(claims(nbf=now + 31)), "not yet valid")


def test_alg_none_is_rejected():
    _rejected(_verifier(), raw_token(claims(), alg="none"), "algorithm")


def test_hs256_signed_with_the_public_key_is_rejected():
    """The classic algorithm swap: a verifier that let the token pick HMAC would key it with the RSA public key, which
    anyone can fetch from the JWKS."""
    _rejected(_verifier(), raw_token(claims(), alg="HS256", key=public_pem(rsa_key())), "algorithm")


def test_an_es256_header_over_an_rsa_key_is_rejected():
    _rejected(_verifier(), mint(claims(), ec_key(), alg="ES256", kid="k1"), "")


def test_a_token_naming_no_key_is_rejected():
    _rejected(_verifier(), mint(claims(), kid=None), "signing key")


def test_a_missing_sub_is_rejected():
    _rejected(_verifier(), mint(claims(sub=None)), "sub")


def test_a_malformed_token_is_rejected():
    _rejected(_verifier(), "not-a-jwt", "malformed")


@pytest.mark.parametrize("audience, refuse", [(AUDIENCE, OWUI_AUDIENCE), (OWUI_AUDIENCE, AUDIENCE)])
def test_a_token_carrying_both_audiences_is_refused_on_either_door(audience, refuse):
    """PyJWT passes an audience found anywhere in the list, so a token minted with both would open both doors."""
    token = mint(claims(aud=[AUDIENCE, OWUI_AUDIENCE]))
    rejected = _rejected(_verifier(audience=audience, refuse=refuse), token, "both")
    assert rejected.sub == "user-1" and rejected.jti == "jti-1"


def test_the_open_webui_door_takes_its_own_audience_only():
    v = _verifier(audience=OWUI_AUDIENCE, refuse=AUDIENCE)
    assert v.claims(mint(claims(aud=OWUI_AUDIENCE)))["sub"] == "user-1"
    _rejected(v, mint(claims(aud=AUDIENCE)), "audience")


def test_an_unknown_kid_refetches_once_then_waits():
    fetch, clock = _Fetch(jwks_for((rsa_key(), "k1"))), _Clock()
    v = _verifier(fetch, clock=clock)
    v.claims(mint(claims()))
    assert fetch.calls == 1
    clock.now += 61
    _rejected(v, mint(claims(), rsa_key("b"), kid="k2"), "unknown signing key")
    assert fetch.calls == 2  # one refetch for the unknown kid
    _rejected(v, mint(claims(), rsa_key("b"), kid="k3"), "unknown signing key")
    assert fetch.calls == 2  # a second unknown kid within 60 s of that fetch waits
    clock.now += 61
    _rejected(v, mint(claims(), rsa_key("b"), kid="k3"), "unknown signing key")
    assert fetch.calls == 3
    assert v.claims(mint(claims()))["sub"] == "user-1"  # the known key never waited


def test_a_rotated_key_verifies_after_one_refetch():
    fetch, clock = _Fetch(jwks_for((rsa_key(), "k1"))), _Clock()
    v = _verifier(fetch, clock=clock)
    v.claims(mint(claims()))
    clock.now += 61
    fetch.jwks = jwks_for((rsa_key("b"), "k2"))
    assert v.claims(mint(claims(), rsa_key("b"), kid="k2"))["sub"] == "user-1"
    assert fetch.calls == 2


def test_an_unreachable_jwks_denies_and_logs(caplog):
    from jwt import PyJWKClientConnectionError

    v = _verifier(_Fetch(error=PyJWKClientConnectionError("connection refused")))
    with caplog.at_level(logging.WARNING, logger="mnemiq.auth"):
        _rejected(v, mint(claims()), "signing keys unavailable")
    assert "JWKS" in caplog.text


def test_a_failed_fetch_is_retried_only_after_the_interval():
    from jwt import PyJWKClientConnectionError

    fetch, clock = _Fetch(error=PyJWKClientConnectionError("down")), _Clock()
    v = _verifier(fetch, clock=clock)
    _rejected(v, mint(claims()), "signing keys unavailable")
    _rejected(v, mint(claims()), "signing keys unavailable")
    assert fetch.calls == 1
    clock.now += 61
    fetch.error, fetch.jwks = None, jwks_for((rsa_key(), "k1"))
    assert v.claims(mint(claims()))["sub"] == "user-1"


def test_the_default_fetch_has_a_five_second_timeout():
    from mnemiq.auth.token import JwtIdentityVerifier

    v = JwtIdentityVerifier(_oidc(), audience=AUDIENCE, refuse_audience=None)
    assert v.fetch_timeout_s == 5.0
    assert v._jwks_client().timeout == 5.0


def test_a_hanging_jwks_is_cut_at_the_timeout():
    """A JWKS endpoint that accepts the connection and never answers must deny, not hang every request."""
    from mnemiq.auth.token import JwtIdentityVerifier

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    held = []
    threading.Thread(target=lambda: held.append(server.accept()), daemon=True).start()
    url = f"http://127.0.0.1:{server.getsockname()[1]}/certs"
    v = JwtIdentityVerifier(_oidc(jwks_url=url), audience=AUDIENCE, refuse_audience=None, fetch_timeout_s=0.5)
    started = time.monotonic()
    _rejected(v, mint(claims()), "signing keys unavailable")
    assert time.monotonic() - started < 3
    server.close()


def test_the_jwks_file_replaces_the_fetch(tmp_path):
    import json

    from mnemiq.auth.token import JwtIdentityVerifier

    path = tmp_path / "jwks.json"
    path.write_text(json.dumps(jwks_for((rsa_key(), "k1"))))
    v = JwtIdentityVerifier(_oidc(jwks_url=None, jwks_file=str(path)), audience=AUDIENCE, refuse_audience=None,
                            fetch=_Fetch(error=AssertionError("the network was used")))
    assert v.claims(mint(claims()))["sub"] == "user-1"


def test_an_unreadable_jwks_file_denies(tmp_path, caplog):
    from mnemiq.auth.token import JwtIdentityVerifier

    v = JwtIdentityVerifier(_oidc(jwks_url=None, jwks_file=str(tmp_path / "missing.json")), audience=AUDIENCE,
                            refuse_audience=None)
    with caplog.at_level(logging.WARNING, logger="mnemiq.auth"):
        _rejected(v, mint(claims()), "signing keys unavailable")
    assert "JWKS" in caplog.text


def test_verify_token_returns_scopes_and_none_on_rejection():
    v = _verifier()
    access = anyio.run(v.verify_token, mint(claims()))
    assert access.subject == "user-1" and access.scopes == ["openid", "db.read"]
    assert access.claims["tenant"] == "acme"
    assert anyio.run(v.verify_token, mint(claims(aud="another"))) is None


def test_verify_token_applies_the_extra_check():
    from mnemiq.auth.token import JwtIdentityVerifier, TokenRejected

    def refuse(found):
        raise TokenRejected("tenant mismatch", sub=found["sub"])

    v = JwtIdentityVerifier(_oidc(), audience=AUDIENCE, refuse_audience=None,
                            fetch=_Fetch(jwks_for((rsa_key(), "k1"))), check=refuse)
    assert anyio.run(v.verify_token, mint(claims())) is None


def test_the_token_is_never_logged(caplog):
    v = _verifier()
    tokens = [mint(claims(aud="another")), mint(claims(aud=[AUDIENCE, OWUI_AUDIENCE])), mint(claims())]
    with caplog.at_level(logging.DEBUG):
        for token in tokens:
            anyio.run(v.verify_token, token)
    for token in tokens:
        assert token not in caplog.text and token.split(".")[2] not in caplog.text


# --- The key set's lifetime, and a fetch that can never stall a check (final review) -----------------------------


def test_a_known_key_is_refetched_after_its_lifetime():
    fetch, clock = _Fetch(jwks_for((rsa_key(), "k1"))), _Clock()
    v = _verifier(fetch, clock=clock)
    v.claims(mint(claims()))
    clock.now += 299
    v.claims(mint(claims()))
    assert fetch.calls == 1
    clock.now += 2
    v.claims(mint(claims()))
    assert fetch.calls == 2


def test_a_key_removed_from_the_jwks_stops_verifying():
    """How an issuer revokes a compromised key: it stops publishing it."""
    fetch, clock = _Fetch(jwks_for((rsa_key(), "k1"))), _Clock()
    v = _verifier(fetch, clock=clock)
    v.claims(mint(claims()))
    fetch.jwks = jwks_for((rsa_key("b"), "k2"))
    clock.now += 301
    _rejected(v, mint(claims()), "unknown signing key")


def test_an_issuer_outage_denies_once_the_keys_are_too_old():
    from jwt import PyJWKClientConnectionError

    fetch, clock = _Fetch(jwks_for((rsa_key(), "k1"))), _Clock()
    v = _verifier(fetch, clock=clock)
    v.claims(mint(claims()))
    fetch.error = PyJWKClientConnectionError("down")
    clock.now += 301
    assert v.claims(mint(claims()))["sub"] == "user-1"  # a failed refresh keeps a set that is not yet too old
    clock.now += 600
    _rejected(v, mint(claims()), "signing keys unavailable")


def test_a_slow_fetch_never_blocks_a_known_key():
    """Anyone can send a token naming an unknown kid; the fetch it sets off must not hold up anyone else's check."""
    release, fetching = threading.Event(), threading.Event()

    class Slow(_Fetch):
        def __call__(self):
            self.calls += 1
            if self.calls > 1:
                fetching.set()
                release.wait(10)
            return self.jwks

    fetch, clock = Slow(jwks_for((rsa_key(), "k1"))), _Clock()
    v = _verifier(fetch, clock=clock)
    v.claims(mint(claims()))
    clock.now += 61
    stranger = threading.Thread(target=lambda: pytest.raises(Exception, v.claims,
                                                             mint(claims(), rsa_key("b"), kid="k9")))
    stranger.start()
    assert fetching.wait(5)
    started = time.monotonic()
    assert v.claims(mint(claims()))["sub"] == "user-1"
    assert time.monotonic() - started < 0.5
    release.set()
    stranger.join(10)


def test_a_fetch_is_cut_at_its_total_deadline():
    """The client's timeout bounds each socket read; a server trickling bytes must still not hold a check."""
    from mnemiq.auth.token import JwtIdentityVerifier

    class Trickle(_Fetch):
        def __call__(self):
            self.calls += 1
            time.sleep(2)
            return jwks_for((rsa_key(), "k1"))

    v = JwtIdentityVerifier(_oidc(), audience=AUDIENCE, refuse_audience=None, fetch=Trickle(), clock=_Clock(),
                            fetch_timeout_s=0.3)
    started = time.monotonic()
    _rejected(v, mint(claims()), "signing keys unavailable")
    assert time.monotonic() - started < 1.5


def test_a_hung_fetch_is_not_started_twice():
    release = threading.Event()

    class Hung(_Fetch):
        def __call__(self):
            self.calls += 1
            release.wait(10)
            return jwks_for((rsa_key(), "k1"))

    from mnemiq.auth.token import JwtIdentityVerifier

    fetch, clock = Hung(), _Clock()
    v = JwtIdentityVerifier(_oidc(), audience=AUDIENCE, refuse_audience=None, fetch=fetch, clock=clock,
                            fetch_timeout_s=0.2)
    _rejected(v, mint(claims()), "signing keys unavailable")
    clock.now += 61
    _rejected(v, mint(claims()), "signing keys unavailable")
    assert fetch.calls == 1
    release.set()
