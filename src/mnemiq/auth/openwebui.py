"""Open WebUI's signed user header: an HS256 token Open WebUI mints per request with the secret it shares with us.

Service trust, not proof of the user: anyone holding the secret can mint any user, and Open WebUI sends the same
token to every backend it calls, which is why the door also requires the connection key. Its key is its own
secret and nothing else, never anything from the identity provider's JWKS, so an HMAC keyed with a public key
cannot verify here either. Its `email` is unverified (Open WebUI's signup checks no mailbox), so it is never
logged and never used for roles.
"""

from __future__ import annotations

import jwt

from mnemiq.auth.config import OPENWEBUI_ISSUER, OpenWebUISettings
from mnemiq.auth.jwks import INVALID_SUBJECT
from mnemiq.auth.token import TokenRejected


class OpenWebUIVerifier:
    def __init__(self, ow: OpenWebUISettings) -> None:
        self._ow = ow

    def claims(self, token: str) -> dict:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise TokenRejected("malformed token") from exc
        if header.get("alg") != "HS256":
            raise TokenRejected("algorithm not allowed")
        try:
            found = jwt.decode(token, self._ow.secret, algorithms=["HS256"], issuer=OPENWEBUI_ISSUER,
                               leeway=self._ow.leeway_s,
                               options={"require": ["sub", "email", "iat", "exp", "iss"]})
        except jwt.InvalidSignatureError as exc:
            raise TokenRejected("bad signature") from exc
        except jwt.ExpiredSignatureError as exc:
            raise TokenRejected("token expired") from exc
        except jwt.InvalidIssuerError as exc:
            raise TokenRejected("wrong issuer") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise TokenRejected(f"token is missing {exc.claim}") from exc
        except INVALID_SUBJECT as exc:
            raise TokenRejected("token sub is not a non-empty string") from exc
        except (jwt.PyJWTError, ValueError, TypeError) as exc:
            raise TokenRejected("invalid token") from exc
        if not isinstance(found["sub"], str) or not found["sub"].strip():
            raise TokenRejected("token sub is not a non-empty string")
        sub = found["sub"]
        if found["exp"] - found["iat"] > self._ow.max_lifetime_s:
            raise TokenRejected("token lifetime above the maximum", sub=sub)
        if found.get("role") == "pending":
            raise TokenRejected("account pending approval", sub=sub)
        return found
