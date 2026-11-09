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
from concurrent.futures import Future, ThreadPoolExecutor

import jwt

from mnemiq.auth.config import OidcSettings
from mnemiq.auth.jwks import ALGORITHMS, INVALID_SUBJECT, usable_keys

logger = logging.getLogger("mnemiq.auth")

REFETCH_INTERVAL_S = 60.0  # at most one JWKS fetch per interval, so unknown `kid`s cannot make us hammer the issuer
FETCH_TIMEOUT_S = 5.0  # the whole fetch, not each socket read: a server trickling bytes is cut here too
# An issuer revokes a key by no longer publishing it, so a fetched set is refreshed once this old: a removed key stops
# verifying within this plus the interval.
KEYS_REFRESH_AFTER_S = 300.0
# Past this with no successful refresh, every token is denied: an issuer outage ends in denials, never in keys
# trusted for ever.
KEYS_MAX_AGE_S = 900.0


class TokenRejected(Exception):
    """A credential refused. `reason` is composed here, never a library's or the token's own words. `sub` and `jti`
    are set only once the signature has verified, so a log line never repeats what an attacker wrote."""

    def __init__(self, reason: str, *, sub: str | None = None, jti: str | None = None) -> None:
        super().__init__(reason)
        self.reason, self.sub, self.jti = reason, sub, jti


class _KeySet:
    """The issuer's signing keys by `kid`, or read once from a file.

    A fetched set is refreshed when a token names an unknown `kid` or once the set is `KEYS_REFRESH_AFTER_S` old, at
    most once per `REFETCH_INTERVAL_S`, and denied outright once it is `KEYS_MAX_AGE_S` old. The fetch runs on its own
    thread under a deadline for the whole fetch, one at a time, and no check ever waits for it: a token naming a
    random `kid` can start a fetch once a minute, and it must not hold up anyone else's check.
    """

    def __init__(self, fetch: Callable[[], dict] | None, file: str | None, clock: Callable[[], float],
                 timeout_s: float = FETCH_TIMEOUT_S) -> None:
        self._fetch, self._file, self._clock, self._timeout_s = fetch, file, clock, timeout_s
        self._keys: dict[str, jwt.PyJWK] = {}
        self._loaded_file = False
        self._fetched_at: float | None = None
        self._last_attempt: float | None = None
        self._state = threading.Lock()  # guards the fields above, and is never held across a fetch
        self._fetching = threading.Lock()  # one fetch at a time, never waited on
        self._pending: Future | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mnemiq-jwks")

    def key(self, kid: str) -> jwt.PyJWK:
        if self._file is not None:
            with self._state:
                if not self._loaded_file:
                    self._keys = self._read_file()
                    self._loaded_file = True
                found = self._keys.get(kid)
            if found is None:
                raise TokenRejected("unknown signing key")
            return found

        now = self._clock()
        with self._state:
            stale = self._fetched_at is None or now - self._fetched_at >= KEYS_REFRESH_AFTER_S
            due = stale or kid not in self._keys
            allowed = self._last_attempt is None or now - self._last_attempt >= REFETCH_INTERVAL_S
        if due and allowed and self._fetching.acquire(blocking=False):
            try:
                self._refresh(now)
            finally:
                self._fetching.release()
        with self._state:
            if self._fetched_at is None or now - self._fetched_at >= KEYS_MAX_AGE_S:
                raise TokenRejected("signing keys unavailable")
            found = self._keys.get(kid)
        if found is None:
            raise TokenRejected("unknown signing key")
        return found

    def _refresh(self, now: float) -> None:
        with self._state:
            # Checked again under the guard: another request may have passed the same check and fetched since.
            if self._last_attempt is not None and now - self._last_attempt < REFETCH_INTERVAL_S:
                return
            self._last_attempt = now
            if self._pending is not None and not self._pending.done():
                return  # an earlier fetch is still hung: never start a second behind it
            self._pending = future = self._executor.submit(self._fetch)
        try:
            keys = self._parse(future.result(timeout=self._timeout_s))
        except Exception as exc:  # a timeout, a connection or parse failure: logged, never sent to the caller
            logger.warning("JWKS fetch failed (%s); keeping keys fetched in the last %d s, if any",
                           type(exc).__name__, int(KEYS_MAX_AGE_S))
            return
        with self._state:
            self._keys, self._fetched_at = keys, now

    def _read_file(self) -> dict[str, jwt.PyJWK]:
        """Startup refuses a file it cannot use; one that changed since still denies with a reason, never a 500."""
        try:
            with open(self._file, encoding="utf-8") as fh:
                keys = usable_keys(json.load(fh))
        except (OSError, ValueError, jwt.PyJWTError) as exc:
            logger.error("JWKS file %s could not be read (%s); denying every token", self._file, type(exc).__name__)
            raise TokenRejected("signing keys unavailable") from exc
        if not keys:
            logger.error("JWKS file %s holds no usable signing key; denying every token", self._file)
            raise TokenRejected("signing keys unavailable")
        return keys

    @staticmethod
    def _parse(data: dict) -> dict[str, jwt.PyJWK]:
        return usable_keys(data)


def token_scopes(found: dict) -> list[str]:
    """A token's scopes, from its space-separated `scope` claim (RFC 8693, as identity providers send it). Anything
    else is no scopes at all: a list, a number, an absent claim."""
    scope = found.get("scope")
    return scope.split() if isinstance(scope, str) else []


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
                             oidc.jwks_file if oidc.jwks_file else None, clock, fetch_timeout_s)

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
        except INVALID_SUBJECT as exc:
            raise TokenRejected("token sub is not a non-empty string") from exc
        except (jwt.PyJWTError, ValueError, TypeError) as exc:
            raise TokenRejected("invalid token") from exc
        if not isinstance(found["sub"], str) or not found["sub"].strip():
            raise TokenRejected("token sub is not a non-empty string")
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
        return AccessToken(token=token, client_id=str(found.get("azp") or found.get("client_id") or ""),
                           scopes=token_scopes(found),
                           expires_at=int(found["exp"]), subject=str(found["sub"]), claims=found)
