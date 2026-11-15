"""How a `/v1` route answers a refusal, shared by `server/app.py` and `server/superset_build.py`.

`refused` turns a failure Superset, or Superset mode itself, answers (`server/superset_refusals.py`) into its response;
`NotPermitted` is the 403 every `/v1` refusal of a verified caller is, carrying where Sign out ends; `unknown_profile`
is the one sentence a profile the caller may not pick is answered with.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException

from mnemiq.profiles import UnknownProfile
from mnemiq.server.superset_refusals import superset_refusal

__all__ = ["NotPermitted", "refused", "unknown_profile"]

logger = logging.getLogger(__name__)


class NotPermitted(HTTPException):
    """A 403 on /v1: a caller the door has verified but who is not permitted, by the door (another tenant, no
    `db.read`) or, in Superset mode, by Superset (`refused`). Its body carries where Sign out ends at the identity
    provider, beside the unchanged `detail`, so the workbench can offer a Sign out that switches accounts. Not a leak:
    the caller is known, and the URL is the provider's public page. A 401 never carries it."""

    def __init__(self, detail: str, headers: dict | None, end_session_url: str | None) -> None:
        super().__init__(status_code=403, detail=detail, headers=headers)
        self.end_session_url = end_session_url


def refused(exc: Exception, doors) -> HTTPException | None:
    """The response a Superset refusal, or a failure Superset mode answers itself, becomes on /v1, or None for any
    other exception. A 401 asks for the Bearer again, as the door's own does, and a 403 carries where Sign out ends, as
    every /v1 403 does."""
    found = superset_refusal(exc)
    if found is None:
        return None
    log = logger.warning if found.status >= 500 else logger.info
    log("/v1 answered %d for %s: %s", found.status, type(exc).__name__, found.detail)
    if found.status == 401:
        return HTTPException(status_code=401, detail=found.detail, headers={"WWW-Authenticate": "Bearer"})
    if found.status == 403:
        challenge = {"WWW-Authenticate": found.challenge} if found.challenge else None
        return NotPermitted(found.detail, challenge, doors.end_session_url if doors is not None else None)
    return HTTPException(status_code=found.status, detail=found.detail)


def unknown_profile(exc: UnknownProfile) -> HTTPException:
    """The 422 a profile the caller may not pick is answered with: composed from what this caller may pick. Unknown
    and forbidden read alike, and neither repeats the name asked for."""
    return HTTPException(status_code=422, detail=f"unknown profile; you may pick: {exc.offered}")
