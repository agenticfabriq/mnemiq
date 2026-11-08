"""The identity provider's token: verified against its published keys, its issuer, its audience and its lifetime.

One verifier per door, each configured with that door's audience only and refusing a token that also carries the
other door's. The algorithms are pinned here, never taken from the token: a verifier that let the token's `alg`
choose HMAC would key it with the RSA public key, which anyone can fetch from the JWKS.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable

import jwt

from mnemiq.auth.config import OidcSettings

logger = logging.getLogger("mnemiq.auth")

ALGORITHMS = ("RS256", "ES256")
REFETCH_INTERVAL_S = 60.0  # at most one JWKS fetch per interval, so unknown `kid`s cannot make us hammer the issuer
FETCH_TIMEOUT_S = 5.0


class TokenRejected(Exception):
    """A credential refused. `reason` is composed here, never a library's or the token's own words. `sub` and `jti`
    are set only once the signature has verified, so a log line never repeats what an attacker wrote."""

    def __init__(self, reason: str, *, sub: str | None = None, jti: str | None = None) -> None:
        super().__init__(reason)
        self.reason, self.sub, self.jti = reason, sub, jti


class _KeySet:
    """The issuer's signing keys by `kid`, fetched lazily, re-fetched at most once per interval for an unknown
    `kid` or after a failure, or read once from a file."""

    def __init__(self, fetch: Callable[[], dict] | None, file: str | None, clock: Callable[[], float]) -> None:
        self._fetch, self._file, self._clock = fetch, file, clock
        self._keys: dict[str, jwt.PyJWK] = {}
        self._loaded_file = False
        self._last_fetch: float | None = None
        self._lock = threading.Lock()

    def key(self, kid: str) -> jwt.PyJWK:
        with self._lock:
            if kid in self._keys:
                return self._keys[kid]
            if self._file is not None:
                if not self._loaded_file:
                    self._keys = self._parse(self._read_file())
                    self._loaded_file = True
            else:
                now = self._clock()
                if self._last_fetch is None or now - self._last_fetch >= REFETCH_INTERVAL_S:
                    self._last_fetch = now
                    try:
                        self._keys = self._parse(self._fetch())
                    except Exception as exc:  # any fetch or parse failure denies; the reason is logged, not sent
                        logger.warning("JWKS fetch failed (%s); denying until the next fetch", type(exc).__name__)
                        if not self._keys:
                            raise TokenRejected("signing keys unavailable") from exc
                elif not self._keys:
                    raise TokenRejected("signing keys unavailable")
            if kid in self._keys:
                return self._keys[kid]
        raise TokenRejected("unknown signing key")

    def _read_file(self) -> dict:
        try:
            with open(self._file) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("JWKS file %s could not be read (%s); denying every token", self._file, type(exc).__name__)
            raise TokenRejected("signing keys unavailable") from exc

    @staticmethod
    def _parse(data: dict) -> dict[str, jwt.PyJWK]:
        keys = {}
        for entry in jwt.PyJWKSet.from_dict(data).keys:
            if entry.key_id and entry.public_key_use in (None, "sig"):
                keys[entry.key_id] = entry
        return keys


class JwtIdentityVerifier:
    """Verifies one door's Bearer tokens. `claims` is the HTTP API's synchronous check; `verify_token` is the MCP
    SDK's `TokenVerifier`, run in a worker thread because a JWKS fetch blocks."""

    def __init__(self, oidc: OidcSettings, *, audience: str, refuse_audience: str | None,
                 fetch: Callable[[], dict] | None = None, clock: Callable[[], float] = time.monotonic,
                 fetch_timeout_s: float = FETCH_TIMEOUT_S,
                 check: Callable[[dict], object] | None = None) -> None:
        self._oidc, self._audience, self._refuse = oidc, audience, refuse_audience
        self.fetch_timeout_s = fetch_timeout_s
        self._check = check
        self._keys = _KeySet(fetch or (lambda: self._jwks_client().fetch_data()),
                             oidc.jwks_file if oidc.jwks_file else None, clock)

    def _jwks_client(self) -> jwt.PyJWKClient:
        # Caching is _KeySet's, so the client fetches exactly when asked to.
        return jwt.PyJWKClient(self._oidc.jwks_url, cache_jwk_set=False, cache_keys=False,
                               timeout=self.fetch_timeout_s)

    def claims(self, token: str) -> dict:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise TokenRejected("malformed token") from exc
        alg = header.get("alg")
        if alg not in ALGORITHMS:
            raise TokenRejected("algorithm not allowed")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise TokenRejected("token names no signing key")
        key = self._keys.key(kid)
        try:
            found = jwt.decode(token, key.key, algorithms=[alg], audience=self._audience, issuer=self._oidc.issuer,
                               leeway=self._oidc.leeway_s, options={"require": ["exp", "iss", "aud", "sub"]})
        except jwt.ExpiredSignatureError as exc:
            raise TokenRejected("token expired") from exc
        except jwt.ImmatureSignatureError as exc:
            raise TokenRejected("token not yet valid") from exc
        except jwt.InvalidAudienceError as exc:
            raise TokenRejected("wrong audience") from exc
        except jwt.InvalidIssuerError as exc:
            raise TokenRejected("wrong issuer") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise TokenRejected(f"token is missing {exc.claim}") from exc
        except (jwt.PyJWTError, ValueError, TypeError) as exc:
            raise TokenRejected("invalid token") from exc
        sub, jti = str(found["sub"]), found.get("jti") if isinstance(found.get("jti"), str) else None
        audiences = found["aud"] if isinstance(found["aud"], list) else [found["aud"]]
        if self._refuse is not None and self._refuse in audiences:
            raise TokenRejected("token carries both doors' audiences", sub=sub, jti=jti)
        return found

    async def verify_token(self, token: str):
        """The MCP SDK's check: an AccessToken, or None, which the transport answers with a 401."""
        import anyio
        from mcp.server.auth.provider import AccessToken

        def verify() -> dict:
            found = self.claims(token)
            if self._check is not None:
                self._check(found)
            return found

        try:
            found = await anyio.to_thread.run_sync(verify)
        except TokenRejected as exc:
            logger.info("MCP token refused: %s (sub=%r jti=%r)", exc.reason, exc.sub, exc.jti)
            return None
        scope = found.get("scope")
        return AccessToken(token=token, client_id=str(found.get("azp") or found.get("client_id") or ""),
                           scopes=scope.split() if isinstance(scope, str) else [],
                           expires_at=int(found["exp"]), subject=str(found["sub"]), claims=found)
