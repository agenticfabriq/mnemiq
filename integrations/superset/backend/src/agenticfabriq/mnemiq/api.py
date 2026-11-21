"""The token route, `POST /extensions/agenticfabriq/mnemiq/token/` with `{"purpose": "ask" | "build"}`, and the
forwarder, `/extensions/agenticfabriq/mnemiq/v1/<path>`, which a panel calls for every mnemiq request.

A thin Flask wrapper. What the route answers is `token.mint_for`, which takes the signed-in user from `g.user` and
nothing from the body, and refuses (403) an embedded guest and anyone not signed in. Authentication is Superset's own
(`@protect`: the session, or a FAB token), and a state-changing call needs the usual `X-CSRFToken`. The route is
registered while the entrypoint is imported, which is when Superset's extension context is set; imported later it
would land under `/api/v1`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator

from flask import Response, current_app, g, request, stream_with_context
from flask_appbuilder.api import expose, permission_name, protect, safe
from superset_core.rest_api.api import RestApi
from superset_core.rest_api.decorators import api

from agenticfabriq.mnemiq import forward
from agenticfabriq.mnemiq.forward import ANSWER_HEADERS, FORWARD_NOT_CONFIGURED, MAX_BODY, ForwardConfig, forward_for
from agenticfabriq.mnemiq.token import NOT_SIGNED_IN, BadRequest, NotSignedIn, SigningConfig, mint_for, signed_in

logger = logging.getLogger(__name__)


@api(id="token_api", name="mnemiq token", description="Mints the signed-in user's short-lived mnemiq token",
     resource_name="token")
class MnemiqApi(RestApi):
    resource_name = "token"
    class_permission_name = "mnemiq"
    openapi_spec_tag = "mnemiq"
    # Flask-AppBuilder exempts an API's blueprint from CSRF unless the class says otherwise (`BaseApi.csrf_exempt`),
    # and `superset_core.rest_api.api.RestApi` does not; Superset's own APIs do (`BaseSupersetApiMixin`). This route
    # mints a token for whoever's session reaches it, so it needs the session's CSRF token.
    csrf_exempt = False

    @expose("/", methods=("POST",))
    @protect()
    @safe
    @permission_name("read")
    def mint(self) -> Response:
        """Answers {"token", "expires_in"} for the signed-in user. Any other field of the body is ignored. An embedded
        guest, or a request with nobody signed in, gets 403 before anything else is read."""
        # `g.user` is the request's user, an embedded guest's `GuestUser` included: `protect` lets a guest through
        # when its role holds this route's permission.
        if not signed_in(g.user):
            return self.response(403, message=NOT_SIGNED_IN)
        try:
            cfg = SigningConfig.from_mapping(current_app.config)
        except ValueError as exc:  # names the setting; never a key or a path's content
            logger.warning("mnemiq token route is not configured: %s", exc)
            return self.response(503, message=str(exc))
        try:
            return self.response(200, **mint_for(g.user, request.get_json(silent=True), cfg, time.time()))
        except NotSignedIn:
            return self.response(403, message=NOT_SIGNED_IN)
        except BadRequest as exc:
            return self.response(400, message=str(exc))


# What a forwarded answer may be: mnemiq's JSON, or a chat answer's event stream. Anything else (a proxy's page in
# front of mnemiq, say) is not passed on to be rendered at Superset's origin.
_PASSED_ON = ("application/json", "text/event-stream")


def _timed_out(exc: Exception) -> bool:
    """Whether `requests` failed waiting for mnemiq to answer, not to connect: a read timeout, raised as is or, while a
    body is read, wrapped in a ConnectionError (`requests.models.Response.iter_content`)."""
    import requests

    if isinstance(exc, requests.ConnectTimeout):
        return False
    if isinstance(exc, requests.Timeout):
        return True
    seen, cause = set(), exc
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if type(cause).__name__ in ("ReadTimeoutError", "ReadTimeout", "TimeoutError", "timeout"):
            return True
        cause = cause.__cause__ or cause.__context__ or (cause.args[0] if cause.args and
                                                          isinstance(cause.args[0], BaseException) else None)
    return False


def _send(outbound: forward.Outbound) -> forward.Inbound:
    """The request sent to mnemiq with `requests` (in Superset's base requirements): no proxy, `.netrc` or other setting
    read from the environment (`trust_env` off), no redirect followed, each wait for mnemiq bounded by
    `outbound.timeout_s` and the whole answer by `outbound.total_s`, however often mnemiq sends a heartbeat. A JSON
    answer is read here, whole; a stream is passed on chunk by chunk, and cut at the total deadline."""
    import requests
    from urllib3.util import SKIP_HEADER

    deadline = time.monotonic() + outbound.total_s
    session = requests.Session()
    session.trust_env = False
    # No header of the session's own (its user agent, accepted encodings), nor urllib3's default user agent: mnemiq
    # receives the outbound's headers alone, beside the host and length every request carries.
    session.headers.clear()
    headers = {**outbound.headers, "User-Agent": SKIP_HEADER}
    try:
        answer = session.request(outbound.method, outbound.url, headers=headers,
                                 data=outbound.body or None, allow_redirects=False, stream=True,
                                 timeout=(5, outbound.timeout_s))
    except requests.RequestException as exc:
        session.close()
        raise (forward.TimedOut() if _timed_out(exc) else forward.Unreachable()) from None
    content_type = answer.headers.get("Content-Type") or ""
    if not content_type.startswith(_PASSED_ON):
        answer.close()
        session.close()
        logger.warning("mnemiq extension: mnemiq answered a forwarded request with something other than JSON or an "
                       "event stream")
        raise forward.Unreachable()

    def read() -> Iterator[bytes]:
        for chunk in answer.iter_content(chunk_size=None):
            if time.monotonic() > deadline:
                raise forward.TimedOut()
            yield chunk

    if not outbound.stream:
        try:
            body = b"".join(read())
        except forward.TimedOut:
            raise
        except requests.RequestException as exc:
            raise (forward.TimedOut() if _timed_out(exc) else forward.Unreachable()) from None
        finally:
            answer.close()
            session.close()
        return forward.Inbound(answer.status_code, content_type, answer.headers.get("WWW-Authenticate"), iter([body]))

    def streamed() -> Iterator[bytes]:
        try:
            yield from read()
        except forward.TimedOut:
            logger.warning("mnemiq extension: a forwarded stream ran past its deadline and was cut")
        except requests.RequestException:
            logger.warning("mnemiq extension: a forwarded stream broke before mnemiq ended it")
        finally:
            answer.close()
            session.close()

    return forward.Inbound(answer.status_code, content_type, answer.headers.get("WWW-Authenticate"), streamed())


def _answer(status: int, sentence: str) -> Response:
    """One of the forwarder's own refusals, as `forward_for` answers its others: `{"detail": ...}`, uncached."""
    return Response(json.dumps({"detail": sentence}), status=status, content_type="application/json",
                    headers=dict(ANSWER_HEADERS))


@api(id="forward_api", name="mnemiq forward", description="Forwards a panel's requests to mnemiq as the signed-in user",
     resource_name="v1")
class MnemiqForward(RestApi):
    resource_name = "v1"
    class_permission_name = "mnemiq"
    openapi_spec_tag = "mnemiq"
    # As the token route: a state-changing request needs the session's CSRF token.
    csrf_exempt = False

    @expose("/<path:rest>", methods=("GET", "POST"))
    @protect()
    @safe
    @permission_name("read")
    def forward(self, rest: str) -> Response:
        """mnemiq's answer to the request, as `forward.forward_for` decides and sends it for the signed-in user. An
        embedded guest or nobody signed in is refused before any setting or the body is read."""
        if not signed_in(g.user):
            return _answer(403, NOT_SIGNED_IN)
        try:
            target = ForwardConfig.from_mapping(current_app.config)
        except ValueError as exc:  # names the setting; never its value
            logger.warning("mnemiq extension: the forwarder is not configured: %s", exc)
            return _answer(503, FORWARD_NOT_CONFIGURED)
        try:
            signing = SigningConfig.from_mapping(current_app.config)
        except ValueError as exc:
            logger.warning("mnemiq extension: the forwarder cannot mint, its token settings are not configured: %s",
                           exc)
            return _answer(503, str(exc))
        body = b""
        if request.method == "POST":
            if (request.content_length or 0) > MAX_BODY:
                body = b"x" * (MAX_BODY + 1)  # refused by its size; nothing of it is read
            else:
                # Read at most one byte past the limit, whatever the request says of its length: a chunked body
                # declares none.
                body = request.stream.read(MAX_BODY + 1)
        found = forward_for(g.user, request.method, f"/v1/{rest}", request.args.to_dict(flat=False), body,
                            signing=signing, target=target, now=time.time(), send=_send)
        headers = dict(ANSWER_HEADERS)
        if found.www_authenticate:
            headers["WWW-Authenticate"] = found.www_authenticate
        if found.content_type.startswith("text/event-stream"):
            headers["X-Accel-Buffering"] = "no"
            return Response(stream_with_context(found.chunks), status=found.status, content_type=found.content_type,
                            headers=headers)
        return Response(b"".join(found.chunks), status=found.status, content_type=found.content_type,
                        headers=headers)
