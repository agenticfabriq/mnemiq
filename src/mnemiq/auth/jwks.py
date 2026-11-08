"""Reading a JWKS: the signing keys in it that a token can name by `kid`. Shared by the startup check on a JWKS file
and by the verifier, so what starts the server is what verifies a token."""

from __future__ import annotations

import jwt
from cryptography.hazmat.primitives.asymmetric import ec, rsa

# The only algorithms a token may be signed with: the verifier checks a token's `alg` against these, and a key counts
# only where it verifies one of them.
ALGORITHMS = ("RS256", "ES256")

# PyJWT's refusal of a non-string `sub`, where this PyJWT has one (it lives in `jwt.exceptions`, and the declared
# floor may predate it); an empty tuple catches nothing, and the verifiers check `sub` themselves either way.
INVALID_SUBJECT = getattr(jwt.exceptions, "InvalidSubjectError", ())


def usable_keys(data) -> dict[str, jwt.PyJWK]:
    """The set's signing keys by `kid`: an RSA public key for RS256, a P-256 public key for ES256. Raises
    `jwt.PyJWTError` for a set PyJWT cannot read. A key with no string `kid`, one marked for encryption, and one
    published for another algorithm (its JWK `alg`, or what PyJWT reads its type as when it names none) are left out,
    so a set of only those starts nothing."""
    keys = {}
    for entry in jwt.PyJWKSet.from_dict(data).keys:
        kid = entry.key_id
        if isinstance(kid, str) and kid and entry.public_key_use in (None, "sig") and _verifies_a_pinned_algorithm(entry):
            keys[kid] = entry
    return keys


def _verifies_a_pinned_algorithm(entry: jwt.PyJWK) -> bool:
    key = entry.key
    if entry.algorithm_name == "RS256":
        return isinstance(key, rsa.RSAPublicKey)
    if entry.algorithm_name == "ES256":
        return isinstance(key, ec.EllipticCurvePublicKey) and isinstance(key.curve, ec.SECP256R1)
    return False
