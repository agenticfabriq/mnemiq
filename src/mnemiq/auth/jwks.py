"""Reading a JWKS: the signing keys in it that a token can name by `kid`. Shared by the startup check on a JWKS file
and by the verifier, so what starts the server is what verifies a token."""

from __future__ import annotations

import jwt

# PyJWT's refusal of a non-string `sub`, where this PyJWT has one (it lives in `jwt.exceptions`, and the declared
# floor may predate it); an empty tuple catches nothing, and the verifiers check `sub` themselves either way.
INVALID_SUBJECT = getattr(jwt.exceptions, "InvalidSubjectError", ())


def usable_keys(data) -> dict[str, jwt.PyJWK]:
    """The set's signing keys by `kid`. Raises `jwt.PyJWTError` for a set PyJWT cannot read; a key with no `kid`,
    or one marked for encryption, is left out."""
    keys = {}
    for entry in jwt.PyJWKSet.from_dict(data).keys:
        if entry.key_id and entry.public_key_use in (None, "sig"):
            keys[entry.key_id] = entry
    return keys
