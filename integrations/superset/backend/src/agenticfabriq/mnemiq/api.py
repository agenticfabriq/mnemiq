"""The token route, `POST /extensions/agenticfabriq/mnemiq/token/` with `{"purpose": "ask" | "build"}`, and the
forwarder, `/extensions/agenticfabriq/mnemiq/v1/<path>`, which a panel calls for every mnemiq request.

A thin Flask wrapper. What the route answers is `token.mint_for`, which takes the signed-in user from `g.user` and
nothing from the body, and refuses (403) an embedded guest and anyone not signed in. Authentication is Superset's own
(`@protect`: the session, or a FAB token), and a state-changing call needs the usual `X-CSRFToken`. The route is
registered while the entrypoint is imported, which is when Superset's extension context is set; imported later it
would land under `/api/v1`.
"""

from __future__ import annotations

import logging
import time

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


def _send(outbound: forward.Outbound) -> forward.Inbound:
    """The request sent to mnemiq with `requests` (in Superset's base requirements), never following a redirect; a
    stream's chunks read as they arrive, each within the idle time."""
    import requests

    try:
        answer = requests.request(outbound.method, outbound.url, headers=outbound.headers,
                                  data=outbound.body or None, allow_redirects=False, stream=outbound.stream,
                                  timeout=(5, outbound.timeout_s))
    except requests.Timeout:
        raise forward.TimedOut() from None
    except requests.RequestException:
        raise forward.Unreachable() from None

    def chunks():
        try:
            yield from answer.iter_content(chunk_size=None)
        finally:
            answer.close()

    return forward.Inbound(answer.status_code, answer.headers.get("Content-Type") or "application/json",
                           answer.headers.get("WWW-Authenticate"),
                           chunks() if outbound.stream else iter([answer.content]))


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
        """mnemiq's answer to the request, as `forward.forward_for` decides and sends it for the signed-in user."""
        try:
            target = ForwardConfig.from_mapping(current_app.config)
        except ValueError as exc:  # names the setting; never its value
            logger.warning("mnemiq extension: the forwarder is not configured: %s", exc)
            return self.response(503, message=FORWARD_NOT_CONFIGURED)
        try:
            signing = SigningConfig.from_mapping(current_app.config)
        except ValueError as exc:
            logger.warning("mnemiq token route is not configured: %s", exc)
            return self.response(503, message=str(exc))
        body = b""
        if request.method == "POST":
            if (request.content_length or 0) > MAX_BODY:
                body = b"x" * (MAX_BODY + 1)  # refused by size before anything is read
            else:
                body = request.get_data(cache=False)[: MAX_BODY + 1]
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
