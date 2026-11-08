"""Keys, key sets and tokens for the identity tests: a local RSA and EC key, their JWKS, and token minters.

Built here rather than fetched, so a test controls every byte the verifier sees, including tokens no real issuer
would mint (`alg: none`, HS256 keyed with a public key).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from functools import lru_cache

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

ISSUER = "https://idp.example/realms/r"
AUDIENCE = "mnemiq-api"
OWUI_AUDIENCE = "mnemiq-openwebui"


@lru_cache(maxsize=None)
def rsa_key(name: str = "a"):
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@lru_cache(maxsize=None)
def ec_key(name: str = "a"):
    return ec.generate_private_key(ec.SECP256R1())


def jwk(key, kid: str) -> dict:
    """The public JWK of `key`, as an identity provider publishes it."""
    if isinstance(key, rsa.RSAPrivateKey):
        found = RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
        found["alg"] = "RS256"
    else:
        found = ECAlgorithm.to_jwk(key.public_key(), as_dict=True)
        found["alg"] = "ES256"
    return {**found, "kid": kid, "use": "sig"}


def jwks_for(*keys: tuple) -> dict:
    """A key set from (key, kid) pairs."""
    return {"keys": [jwk(key, kid) for key, kid in keys]}


def public_pem(key) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def claims(**over) -> dict:
    """A valid identity-provider token's claims, with `over` replacing or (as None) removing any of them."""
    now = int(time.time())
    base = {"iss": ISSUER, "aud": AUDIENCE, "sub": "user-1", "iat": now, "nbf": now, "exp": now + 300,
            "jti": "jti-1", "tenant": "acme", "realm_access": {"roles": ["analyst"]}, "scope": "openid db.read"}
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


def mint(body: dict, key=None, *, alg: str = "RS256", kid: str | None = "k1", headers: dict | None = None) -> str:
    key = rsa_key() if key is None else key
    head = {**({"kid": kid} if kid is not None else {}), **(headers or {})}
    return jwt.encode(body, key, algorithm=alg, headers=head)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def raw_token(body: dict, *, alg: str, key: bytes | None = None, kid: str | None = "k1") -> str:
    """A token signed by hand: `alg: none` with no signature, or an HMAC over any bytes at all, which is what an
    attacker does when PyJWT would refuse to encode it (an HMAC keyed with a public key)."""
    head = {"alg": alg, "typ": "JWT", **({"kid": kid} if kid is not None else {})}
    signing_input = f"{_b64(json.dumps(head).encode())}.{_b64(json.dumps(body).encode())}"
    if alg == "none":
        return signing_input + "."
    digest = {"HS256": hashlib.sha256}[alg]
    return signing_input + "." + _b64(hmac.new(key, signing_input.encode(), digest).digest())


def hs256(body: dict, secret: bytes, headers: dict | None = None) -> str:
    return jwt.encode(body, secret, algorithm="HS256", headers=headers)


def owui_claims(**over) -> dict:
    """A valid Open WebUI signed-header token's claims, as v0.13.0 mints them."""
    now = int(time.time())
    base = {"sub": "owui-user-1", "email": "ana@example.com", "name": "Ana", "role": "user", "iss": "open-webui",
            "iat": now, "exp": now + 300}
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}
