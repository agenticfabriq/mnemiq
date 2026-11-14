"""Mints the short-lived token mnemiq presents to Superset's MCP service for the person signed in.

Pure on purpose: no Flask and no Superset import at module level, so the minting and the request handler behind the
route (`mint_for`) run, and are tested, without either. `api.py` is the thin Flask wrapper.

The signing key is a file the operator provides, readable by its owner only. The token is RS256 with a `kid`, so
mnemiq's door and Superset's MCP verifier check it against the public half they hold. It lives at most 300 seconds.

A token is minted only for a user signed in to Superset itself (`signed_in`), never for an embedded guest or for nobody
at all: the MCP service resolves a token to the Superset user of its name, and a guest's name is whatever its guest
token says.
"""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import jwt

Purpose = Literal["ask", "build"]

MAX_LIFETIME_S = 300
# How far back `nbf` is set. Superset's MCP verifier refuses a token whose `nbf` is after its own clock, with no leeway
# (`superset/mcp_service/jwt_verifier.py`, the not-before check), and the token is minted by the web server, often
# another process or machine: a token used the moment it is minted must not fail on a clock a second behind.
NOT_BEFORE_SKEW_S = 5

# What an ask token carries: the scopes the existing Superset mode asks with.
ASK_SCOPES: tuple[str, ...] = ("db.read", "superset:sqllab:write", "superset:dataset:read")
# A build token adds what building a dashboard writes: charts, the dashboard, and datasets.
BUILD_SCOPES: tuple[str, ...] = ASK_SCOPES + (
    "superset:chart:read", "superset:chart:write", "superset:dashboard:read", "superset:dashboard:write",
    "superset:dataset:write")
SCOPES: dict[str, tuple[str, ...]] = {"ask": ASK_SCOPES, "build": BUILD_SCOPES}

_REQUIRED = ("MNEMIQ_TOKEN_KEY_FILE", "MNEMIQ_TOKEN_KID", "MNEMIQ_TOKEN_ISSUER", "MNEMIQ_TOKEN_AUDIENCE",
             "MNEMIQ_TOKEN_TENANT")


class BadRequest(ValueError):
    """The route's body is not one it answers: a 400."""


NOT_SIGNED_IN = "the token route mints only for a signed-in Superset user"


class NotSignedIn(PermissionError):
    """The request's user is not one the route mints for (`signed_in`): a 403."""

    def __init__(self) -> None:
        super().__init__(NOT_SIGNED_IN)


def signed_in(user) -> bool:
    """Whether `user` is a user signed in to Superset itself: not an embedded guest, not anonymous, and with a positive
    integer id, as Flask-AppBuilder's `User` model is (`flask_appbuilder/security/sqla/models.py`).

    - An embedded guest is refused whatever else it says. With `EMBEDDED_SUPERSET` on, a request carrying a guest
      token is run as a `GuestUser` (`superset/security/guest_token.py`; `request_loader` and
      `get_guest_user_from_token` in `superset/security/manager.py`), which is authenticated and not anonymous, has
      no id, and has the `username` the embedding host wrote into the guest token; its role (`GUEST_ROLE_NAME`) may
      hold this route's permission. A deployment's security manager may replace the class
      (`SupersetSecurityManager.guest_user_cls`), so the guest check does not rest on the missing id.
    - Nobody signed in (flask_login's `AnonymousUserMixin`, `is_anonymous` True) reaches the route when the Public
      role holds its permission (`PUBLIC_ROLE_LIKE`, `is_item_public` in Flask-AppBuilder's `protect`).
    - A user without an id is no row of Superset's users table: no MCP lookup of its name could be its own."""
    if getattr(user, "is_guest_user", False):
        return False
    if getattr(user, "is_anonymous", True):
        return False
    user_id = getattr(user, "id", None)
    return isinstance(user_id, int) and not isinstance(user_id, bool) and user_id > 0


@dataclass(frozen=True)
class SigningConfig:
    key_pem: bytes = field(repr=False)
    kid: str
    issuer: str
    audience: str
    tenant: str
    lifetime_s: int = MAX_LIFETIME_S

    @classmethod
    def from_mapping(cls, config: Mapping) -> SigningConfig:
        """Reads the settings from Superset's config (any mapping). Refuses a missing one, a non-positive lifetime,
        and a key that is not a regular file readable by its owner only."""
        found: dict[str, str] = {}
        for name in _REQUIRED:
            value = config.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is not set")
            found[name] = value.strip()
        lifetime = config.get("MNEMIQ_TOKEN_LIFETIME_S", MAX_LIFETIME_S)
        try:
            lifetime = int(lifetime)
        except (TypeError, ValueError):
            raise ValueError("MNEMIQ_TOKEN_LIFETIME_S must be a whole number of seconds") from None
        if lifetime < 1:
            raise ValueError("MNEMIQ_TOKEN_LIFETIME_S must be at least 1")
        return cls(key_pem=_read_key(found["MNEMIQ_TOKEN_KEY_FILE"]), kid=found["MNEMIQ_TOKEN_KID"],
                   issuer=found["MNEMIQ_TOKEN_ISSUER"], audience=found["MNEMIQ_TOKEN_AUDIENCE"],
                   tenant=found["MNEMIQ_TOKEN_TENANT"], lifetime_s=min(lifetime, MAX_LIFETIME_S))


def _read_key(path: str) -> bytes:
    """The key's bytes, from a regular file (not a link) with mode 0600, judged on the open descriptor itself."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"MNEMIQ_TOKEN_KEY_FILE must be a regular file mnemiq can read: {exc.strerror}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("MNEMIQ_TOKEN_KEY_FILE must be a regular file")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("MNEMIQ_TOKEN_KEY_FILE must have mode 0600: the signing key is for its owner only")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def mint(cfg: SigningConfig, *, username: str, purpose: Purpose, now: float) -> str:
    if purpose not in SCOPES:
        raise ValueError(f"purpose must be one of {sorted(SCOPES)}")
    if not isinstance(username, str) or not username:
        raise ValueError("a token needs a username")
    issued = int(now)
    claims = {"iss": cfg.issuer, "aud": cfg.audience, "sub": username, "preferred_username": username,
              "iat": issued, "nbf": issued - NOT_BEFORE_SKEW_S, "exp": issued + cfg.lifetime_s,
              "jti": secrets.token_urlsafe(16),
              "tenant": cfg.tenant, "scope": " ".join(SCOPES[purpose])}
    return jwt.encode(claims, cfg.key_pem, algorithm="RS256", headers={"kid": cfg.kid})


def mint_for(user, body, cfg: SigningConfig, now: float) -> dict:
    """What the route answers. The token is for `user.username` and nobody else: the body names a purpose and is
    read for nothing more, so a username in it changes nothing. Raises `NotSignedIn` for a user `signed_in` refuses,
    before the body is read, and `BadRequest` for a body without a purpose."""
    if not signed_in(user):
        raise NotSignedIn()
    purpose = body.get("purpose") if isinstance(body, dict) else None
    if not isinstance(purpose, str) or purpose not in SCOPES:
        raise BadRequest('the body must be {"purpose": "ask"} or {"purpose": "build"}')
    return {"token": mint(cfg, username=user.username, purpose=purpose, now=now), "expires_in": cfg.lifetime_s}
