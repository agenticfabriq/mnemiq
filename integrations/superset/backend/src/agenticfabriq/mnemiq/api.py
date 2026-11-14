"""The token route: `POST /extensions/agenticfabriq/mnemiq/token/` with `{"purpose": "ask" | "build"}`.

A thin Flask wrapper. What the route answers is `token.mint_for`, which takes the signed-in user from `g.user` and
nothing from the body. Authentication is Superset's own (`@protect`: the session, or a FAB token), and a state-changing
call needs the usual `X-CSRFToken`. The route is registered while the entrypoint is imported, which is when
Superset's extension context is set; imported later it would land under `/api/v1`.
"""

from __future__ import annotations

import logging
import time

from flask import Response, current_app, g, request
from flask_appbuilder.api import expose, permission_name, protect, safe
from superset_core.rest_api.api import RestApi
from superset_core.rest_api.decorators import api

from agenticfabriq.mnemiq.token import BadRequest, SigningConfig, mint_for

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
        """Answers {"token", "expires_in"} for the signed-in user. Any other field of the body is ignored."""
        try:
            cfg = SigningConfig.from_mapping(current_app.config)
        except ValueError as exc:  # names the setting; never a key or a path's content
            logger.warning("mnemiq token route is not configured: %s", exc)
            return self.response(503, message=str(exc))
        try:
            return self.response(200, **mint_for(g.user, request.get_json(silent=True), cfg, time.time()))
        except BadRequest as exc:
            return self.response(400, message=str(exc))
