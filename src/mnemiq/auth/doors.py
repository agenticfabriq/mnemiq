"""The credential rules of the HTTP doors: which credential each door accepts, and why any other is refused.

A request is judged on exactly one credential. Two are refused, never settled by precedence, and a credential that
fails is refused without trying another kind. `/v1` takes the identity provider's token for its own audience;
`/openai/v1` takes Open WebUI's signed header, or a provider token for its own, separate audience, and with a
connection key configured it also needs that key. Open WebUI's unsigned user headers are refused on every door:
their presence means Open WebUI is forwarding identity without signing it, and answering would hide that.

Free of the web framework on purpose: `server/app.py` hands in the headers and turns a refusal into its response.
Every reason is composed here, and a refusal's log line carries the reason, `sub` and `jti` only, never a token,
an email or the connection key.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Callable, Mapping

import jwt

from mnemiq.auth.config import AuthConfig
from mnemiq.auth.identity_map import IdentityMapUnavailable
from mnemiq.auth.openwebui import OpenWebUIVerifier
from mnemiq.auth.resolve import TenantMismatch, identity_from_oidc, identity_from_openwebui
from mnemiq.auth.token import JwtIdentityVerifier, TokenRejected, token_scopes
from mnemiq.contract import IdentityContext

logger = logging.getLogger("mnemiq.auth")

Headers = Mapping[str, list[str]]  # every value of each header, keyed by its lowercase name

V1_SCOPE = "db.read"  # what /v1 requires of a token, as MCP's read tools do


class DoorRefusal(Exception):
    """`challenge` is the `WWW-Authenticate` value for a refusal that carries one beyond the plain `Bearer` a 401
    sends: an insufficient scope's (RFC 6750)."""

    def __init__(self, status: int, reason: str, *, challenge: str | None = None) -> None:
        super().__init__(reason)
        self.status, self.reason, self.challenge = status, reason, challenge


class HttpDoors:
    def __init__(self, cfg: AuthConfig, *, fetch: Callable[[], dict] | None = None) -> None:
        self._cfg = cfg
        oidc = cfg.oidc
        # One verifier per door, each holding its own audience only and refusing the other's.
        self._v1 = (JwtIdentityVerifier(oidc, audience=oidc.audience, refuse_audience=oidc.openwebui_audience,
                                        fetch=fetch) if oidc else None)
        self._openai = (JwtIdentityVerifier(oidc, audience=oidc.openwebui_audience, refuse_audience=oidc.audience,
                                            fetch=fetch) if oidc and oidc.openwebui_audience else None)
        self._owui = OpenWebUIVerifier(cfg.openwebui) if cfg.openwebui else None

    @property
    def openai_enabled(self) -> bool:
        """Whether /openai/v1 can verify a caller: Open WebUI's signed header, or a provider token for its own
        audience. Without either the endpoint is not mounted, so a deployment cannot serve it as one identity."""
        return self._owui is not None or self._openai is not None

    @property
    def task_header(self) -> str:
        """The header (lowercase) whose non-empty value marks an Open WebUI background task on /openai/v1."""
        return self._cfg.task_header

    def v1(self, headers: Headers) -> IdentityContext:
        return self._judge("/v1", headers)

    def openai(self, headers: Headers) -> IdentityContext:
        return self._judge("/openai/v1", headers)

    def _judge(self, door: str, headers: Headers) -> IdentityContext:
        try:
            return self._identity(door, headers)
        except DoorRefusal as exc:
            logger.info("refused on %s: %s", door, exc.reason)
            raise
        except TokenRejected as exc:
            status = 503 if isinstance(exc, IdentityMapUnavailable) else 403 if isinstance(exc, TenantMismatch) else 401
            log = logger.warning if status == 503 else logger.info
            log("refused on %s: %s (sub=%r jti=%r)", door, exc.reason, exc.sub, exc.jti)
            raise DoorRefusal(status, exc.reason) from None

    def _identity(self, door: str, headers: Headers) -> IdentityContext:
        cfg = self._cfg
        bearer, signed = screen(cfg, door, headers)
        if door == "/openai/v1":
            if bearer is not None and signed:
                raise DoorRefusal(401, _two_credentials(bearer))
            if cfg.binding is not None:
                name, key = cfg.binding
                sent = headers.get(name, [])
                # Constant time, and on bytes: the key is a secret and the comparison must not leak its prefix.
                if len(sent) != 1 or not hmac.compare_digest(sent[0].encode("latin-1", "replace"), key):
                    raise DoorRefusal(401, "request did not come through mnemiq's connection")
            if signed:
                if self._owui is None:
                    raise DoorRefusal(401, "the Open WebUI header is not enabled on this deployment")
                return identity_from_openwebui(self._owui.claims(signed[0]), cfg)

        if bearer is not None:
            verifier = self._v1 if door == "/v1" else self._openai
            if verifier is None:
                if door == "/v1":
                    raise DoorRefusal(401, "no identity provider is configured to verify a Bearer token on /v1")
                raise DoorRefusal(401, "no Bearer token is accepted here without MNEMIQ_OPENWEBUI_AUDIENCE")
            found = verifier.claims(bearer)
            identity = identity_from_oidc(found, cfg)
            # A 403, not a 401: the caller is known and not permitted, which signing in again would not change.
            if door == "/v1" and V1_SCOPE not in token_scopes(found):
                raise DoorRefusal(403, f"insufficient scope: this token lacks {V1_SCOPE}",
                                  challenge=f'Bearer error="insufficient_scope", scope="{V1_SCOPE}"')
            return identity
        raise DoorRefusal(401, "no credential")


def screen(cfg: AuthConfig, door: str, headers: Headers) -> tuple[str | None, list[str]]:
    """The checks that come before any credential is verified, the same on every door: Open WebUI's unsigned headers,
    a credential sent twice, a malformed Authorization, and, on any door but /openai/v1, Open WebUI's signed header.
    Returns the Bearer token (or None) and the signed header's values. The MCP door runs this in front of the SDK,
    whose own check reads the first Authorization header and nothing else."""
    signed = headers.get(cfg.jwt_header, [])
    if any(headers.get(name) for name in cfg.user_headers):
        if signed:
            # Open WebUI sends the signed header instead of the plain ones, never beside them, so something
            # else added these: a custom connection header or a proxy.
            raise DoorRefusal(401, "conflicting identity headers: Open WebUI's unsigned user headers arrived "
                                   "beside its signed one")
        raise DoorRefusal(401, "unsigned identity header: Open WebUI's fallback, meaning its JWT minting failed "
                               "or no secret is set")
    authorization = headers.get("authorization", [])
    if len(authorization) > 1 or len(signed) > 1:
        raise DoorRefusal(401, "ambiguous credential: a credential header was sent more than once")
    bearer = _bearer(authorization[0]) if authorization else None
    if signed and door != "/openai/v1":
        raise DoorRefusal(401, "the Open WebUI header is accepted on /openai/v1 only")
    return bearer, signed


def _bearer(value: str) -> str:
    scheme, _, token = value.strip().partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token or any(c.isspace() for c in token):
        raise DoorRefusal(401, "malformed credential: expected `Authorization: Bearer <token>`")
    return token


def _two_credentials(bearer: str) -> str:
    """Which fix to name. The token's shape picks the message and nothing else: both cases are refused."""
    try:
        jwt.get_unverified_header(bearer)
        looks_like_a_token = True
    except jwt.PyJWTError:
        looks_like_a_token = False
    if looks_like_a_token:
        return ("two credentials: an identity provider's token and Open WebUI's signed header; with system_oauth, "
                "turn Open WebUI's user forwarding off")
    return "two credentials: set the connection's auth type to none when forwarding the signed user header"
