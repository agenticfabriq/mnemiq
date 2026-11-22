"""Forwards a panel's requests to mnemiq as the signed-in person: `/extensions/agenticfabriq/mnemiq/v1/<path>`.

Pure on purpose, as `token.py` is: no Flask and no Superset import at module level, so the checks and the request
mnemiq receives are tested without either. `api.py`'s `MnemiqForward` is the thin Flask wrapper and its real `send`.

What it forwards, and how:
- Only the mnemiq routes `ROUTES` lists, each by its method, each path read whole (`re.fullmatch`), with no `.` or
  `..` segment and no encoded `/` in a measure's name. Anything else is refused before mnemiq is called.
- Each request with a token minted for it alone (`token.mint`), for the signed-in user (`token.signed_in`) and for
  the route's purpose: an ask token for asking, a build token for every `/v1/superset/` route. The browser never
  holds a mnemiq token, nor mnemiq's address.
- mnemiq receives exactly `Authorization`, `Accept`, and `Content-Type` when there is a body: never Superset's cookie,
  its CSRF header or any other of the browser's headers, which are not even an input here.
- The answer carries back only mnemiq's status, `Content-Type` and `WWW-Authenticate`, with `Cache-Control: no-store`.
  A chat answer is a stream, passed on chunk by chunk as it arrives.
- Neither a token nor a body is logged.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote, urlencode, urlsplit

from agenticfabriq.mnemiq.token import NOT_SIGNED_IN, SigningConfig, mint, signed_in

logger = logging.getLogger(__name__)

Purpose = Literal["ask", "build"]

MAX_BODY = 65_536

NOT_FORWARDED = "This extension does not forward that request to mnemiq."
BODY_TOO_LARGE = f"The request is larger than this extension forwards to mnemiq ({MAX_BODY:,} bytes at most)."
QUERY_REFUSED = "This extension does not forward that request's query to mnemiq."
MNEMIQ_UNREACHABLE = ("mnemiq did not answer; try again, and if it keeps failing ask your Superset administrator to "
                      "check mnemiq.")
MNEMIQ_TIMED_OUT = ("mnemiq took too long to answer; try again, and if it keeps happening ask your Superset "
                    "administrator to check mnemiq and the timeouts in front of it.")
FORWARD_NOT_CONFIGURED = ("This Superset's mnemiq extension does not know where mnemiq is: set MNEMIQ_URL in "
                          "Superset's configuration.")
# A token setting missing or not valid: the person is told only that much; Superset's log names the setting.
SIGNING_NOT_CONFIGURED = ("This Superset's mnemiq extension cannot sign its requests to mnemiq: a token setting is "
                          "missing or not valid, and Superset's log names it.")

# The headers every forwarded answer carries, beside mnemiq's `Content-Type` and `WWW-Authenticate`.
ANSWER_HEADERS = {"Cache-Control": "no-store"}

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_INT = r"[1-9][0-9]{0,9}"
# A measure's name: one path segment of the path as Superset's router decoded it, encoded once for mnemiq.
_NAME = r"(?P<name>[^/]{1,250})"
_ONE_ID = re.compile(_INT)


@dataclass(frozen=True)
class Route:
    method: str
    pattern: re.Pattern[str]
    purpose: Purpose
    query: frozenset[str] = frozenset()
    stream: bool = False


ROUTES: tuple[Route, ...] = (
    Route("GET", re.compile(r"/v1/schema"), "ask"),
    Route("POST", re.compile(r"/v1/chat"), "ask", stream=True),
    Route("GET", re.compile(r"/v1/superset/builds"), "build", frozenset({"dashboard_id"})),
    Route("POST", re.compile(r"/v1/superset/builds"), "build"),
    Route("GET", re.compile(rf"/v1/superset/builds/{_UUID}"), "build"),
    Route("POST", re.compile(rf"/v1/superset/builds/{_UUID}/next"), "build"),
    Route("POST", re.compile(rf"/v1/superset/builds/{_UUID}/confirm"), "build"),
    Route("GET", re.compile(rf"/v1/superset/dashboards/{_INT}/measures"), "build"),
    Route("POST", re.compile(rf"/v1/superset/dashboards/{_INT}/measures/{_NAME}/(edit|remove|save)"), "build"),
    Route("POST", re.compile(rf"/v1/superset/dashboards/{_INT}/changes"), "build"),
)
# Each query key a route takes, and what its one value must be.
_QUERY_VALUES: dict[str, re.Pattern[str]] = {"dashboard_id": _ONE_ID}


@dataclass(frozen=True)
class ForwardConfig:
    """Where mnemiq is (`MNEMIQ_URL`, a path prefix kept), how long a forwarded request may take in all
    (`MNEMIQ_FORWARD_TIMEOUT_S`, 1 to 900), and how long a stream may go quiet (`MNEMIQ_FORWARD_IDLE_S`, 5 to 300)."""

    url: str
    timeout_s: float = 600.0
    idle_s: float = 60.0

    @classmethod
    def from_mapping(cls, config: Mapping) -> ForwardConfig:
        raw = config.get("MNEMIQ_URL")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("MNEMIQ_URL is not set")
        parts = urlsplit(raw.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("MNEMIQ_URL must be an http or https address")
        if parts.username is not None or parts.password is not None or "@" in parts.netloc:
            raise ValueError("MNEMIQ_URL must hold no user or password")
        if parts.query or parts.fragment:
            raise ValueError("MNEMIQ_URL must hold no query or fragment")
        url = f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"
        return cls(url=url, timeout_s=_seconds(config, "MNEMIQ_FORWARD_TIMEOUT_S", 600.0, 1, 900),
                   idle_s=_seconds(config, "MNEMIQ_FORWARD_IDLE_S", 60.0, 5, 300))


def _seconds(config: Mapping, name: str, default: float, low: float, high: float) -> float:
    value = config.get(name, default)
    try:
        found = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number of seconds") from None
    if isinstance(value, bool) or not low <= found <= high:
        raise ValueError(f"{name} must be from {low:g} to {high:g} seconds")
    return found


@dataclass(frozen=True)
class Outbound:
    """The request mnemiq is sent. `timeout_s` bounds each wait for mnemiq (to connect, and between two reads: a
    stream's idle time); `total_s` bounds the whole answer, read to its end, however often mnemiq sends something."""

    method: str
    url: str
    headers: dict[str, str] = field(repr=False)
    body: bytes = field(repr=False)
    timeout_s: float
    stream: bool
    total_s: float = 600.0


@dataclass(frozen=True)
class Inbound:
    status: int
    content_type: str
    www_authenticate: str | None
    chunks: Iterator[bytes]


Send = Callable[[Outbound], Inbound]


class Unreachable(Exception):
    """mnemiq could not be reached: a refused connection, a reset, a name that does not resolve."""


class TimedOut(Exception):
    """mnemiq did not answer within the request's time, or a stream went quiet past the idle time."""


def _refusal(status: int, sentence: str) -> Inbound:
    return Inbound(status, "application/json", None, iter([json.dumps({"detail": sentence}).encode()]))


def _name_problem(name: str) -> bool:
    """Whether a measure's name (as decoded once, by Superset's router) is no name: a `.` or `..` segment, or one
    holding an encoded `/` that another decoding could turn into a path."""
    return name in (".", "..") or "%2f" in name.lower()


def _route(method: str, path: str) -> tuple[Route | None, bool, str]:
    """The route `method` and `path` (decoded, as Superset's router hands it over) name, whether some route has the
    path under another method, and the path as mnemiq is sent it (a measure's name encoded once)."""
    # Each pattern is matched whole and none holds `.` or `/` but a measure's name, checked decoded below: a path
    # with a `.` or `..` segment, or two slashes, matches none.
    elsewhere = False
    for route in ROUTES:
        found = route.pattern.fullmatch(path)
        if found is None:
            continue
        if route.method != method:
            elsewhere = True
            continue
        name = found.groupdict().get("name")
        if name is not None:
            if _name_problem(name):
                return None, False, path
            start, end = found.span("name")
            path = f"{path[:start]}{quote(name, safe='')}{path[end:]}"
        return route, False, path
    return None, elsewhere, path


def forward_for(user, method: str, path: str, query: Mapping[str, list[str]], body: bytes, *,
                signing: SigningConfig, target: ForwardConfig, now: float, send: Send) -> Inbound:
    """The answer to the panel's request `method path?query` with `body`, as the signed-in `user`; `path` is decoded,
    as Superset's router hands it over. Checked in this
    order, each refusal answered before mnemiq is called: the user is signed in (403), the path is a route (404),
    under its method (405), the query holds only the route's keys, each once and well formed (400), and the body is
    at most `MAX_BODY` (413). Then a token is minted for the user and the route's purpose, and the request sent."""
    if not signed_in(user):
        return _refusal(403, NOT_SIGNED_IN)
    route, elsewhere, sent_path = _route(method, path)
    if route is None:
        return _refusal(405 if elsewhere else 404, NOT_FORWARDED)
    for key, values in query.items():
        pattern = _QUERY_VALUES.get(key)
        if key not in route.query or pattern is None or len(values) != 1 or not pattern.fullmatch(values[0]):
            return _refusal(400, QUERY_REFUSED)
    if len(body) > MAX_BODY:
        return _refusal(413, BODY_TOO_LARGE)
    token = mint(signing, username=user.username, purpose=route.purpose, now=now)
    headers = {"authorization": f"Bearer {token}",
               "accept": "text/event-stream" if route.stream else "application/json"}
    sent_body = body if method == "POST" else b""
    if sent_body:
        headers["content-type"] = "application/json"
    url = f"{target.url}{sent_path}" + (f"?{urlencode({k: v[0] for k, v in query.items()})}" if query else "")
    outbound = Outbound(method=method, url=url, headers=headers, body=sent_body,
                        timeout_s=target.idle_s if route.stream else target.timeout_s, stream=route.stream,
                        total_s=target.timeout_s)
    try:
        return send(outbound)
    except TimedOut:
        logger.warning("mnemiq extension: a forwarded %s %s timed out", method, route.pattern.pattern)
        return _refusal(504, MNEMIQ_TIMED_OUT)
    except Unreachable:
        logger.warning("mnemiq extension: mnemiq could not be reached for a forwarded %s %s", method,
                       route.pattern.pattern)
        return _refusal(502, MNEMIQ_UNREACHABLE)
