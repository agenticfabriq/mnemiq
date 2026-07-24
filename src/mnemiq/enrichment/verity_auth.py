from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# (token_url, client_id) -> (access_token, expires_at_epoch_seconds)
_TOKEN_CACHE: dict[tuple[str, str], tuple[str, float]] = {}

# Refresh this far ahead of the server's expiry, to absorb clock skew and flight time.
_EXPIRY_MARGIN_SECS = 60
_DEFAULT_TTL_SECS = 600


def reset_token_cache() -> None:
    """Drop every cached token (tests; also the hook for a credential rotation)."""
    _TOKEN_CACHE.clear()


def _credentials(settings) -> tuple[str, str, str] | None:
    """The three settings that make a client-credentials exchange possible, or None."""
    token_url = getattr(settings, "verity_token_url", None)
    client_id = getattr(settings, "verity_client_id", None)
    client_secret = getattr(settings, "verity_client_secret", None)
    if not (token_url and client_id and client_secret):
        return None
    return token_url, client_id, client_secret


def access_token(settings, *, force_refresh: bool = False) -> str | None:
    """Return a cached-or-fresh Verity access token, or None.

    None means "send no Authorization header": either the client credentials are not
    configured (dev, or a Verity running in local auth mode) or acquisition failed.
    Fail-soft is the contract -- a token problem degrades to local-only enrichment and
    never raises into the enrichment pipeline.
    """
    credentials = _credentials(settings)
    if credentials is None:
        return None
    token_url, client_id, client_secret = credentials
    cache_key = (token_url, client_id)

    if not force_refresh:
        cached = _TOKEN_CACHE.get(cache_key)
        if cached is not None and cached[1] > time.time():
            return cached[0]

    body = json.dumps({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()
    request = urllib.request.Request(
        token_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # `exc` carries the URL and status only -- the secret must never reach a log line.
        logger.warning("verity token acquisition failed; enriching local-only: %s", exc)
        _TOKEN_CACHE.pop(cache_key, None)
        return None

    token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        logger.warning("verity token response carried no access_token; enriching local-only")
        _TOKEN_CACHE.pop(cache_key, None)
        return None

    try:
        ttl_secs = int(payload.get("expires_in", _DEFAULT_TTL_SECS))
    except (TypeError, ValueError):
        ttl_secs = _DEFAULT_TTL_SECS
    # A TTL at or under the margin yields expires_at <= now: correct, if chatty -- such a
    # token is re-acquired on every call rather than being used past its usable life.
    _TOKEN_CACHE[cache_key] = (token, time.time() + max(ttl_secs - _EXPIRY_MARGIN_SECS, 0))
    return token
