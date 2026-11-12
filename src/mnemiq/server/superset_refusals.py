"""What a person is told when Superset, or Superset mode itself, refuses a request or cannot serve it.

Every /v1 route and the chat stream read a failure through `superset_refusal`, so no refusal ends as a 500 or as the
stream's opaque error, and each says what the person can do about it, or that nothing they can do is known to help. The
status and the sentence are chosen by the exception's type. Where a message is passed on, it is one mnemiq composed
(`mnemiq.superset.client` keeps Superset's own words in `.detail`, which goes to the log only).

Several of these never reach a route as the code stands, because a layer below catches them first: a statement's
refusal becomes the runner's `ExecutionError`, the function inventory reads a catalogue method that raises as an
answer it could not get, and the grants read a refused probe as "not database-wide". This is the backstop for any
other path.
"""

from __future__ import annotations

from dataclasses import dataclass

from mnemiq.superset.client import (
    SupersetPermissionDenied,
    SupersetRefusal,
    SupersetRefusedCaller,
    SupersetToolError,
    SupersetUnavailable,
)
from mnemiq.superset.executor import SupersetCredentialMissing, SupersetReadOnly
from mnemiq.superset.grants import NOT_A_BUILDER, NotABuilder, SupersetMisconfigured
from mnemiq.superset.payload import SupersetAccessRefused

# Superset no longer accepts the person's token, or knows no user for it.
SIGN_IN_AGAIN = "Superset did not accept this sign-in; sign in to Superset again."
# Superset refused and its words do not say whether the credential or the resource was the cause, so neither signing in
# again nor asking for roles is known to help, and neither is suggested.
DENIED = "Superset refused this request and did not say why; its log records the reason."
ACCESS_REFUSED = "Superset refused access to something this request reads."


@dataclass(frozen=True)
class Refused:
    """The response a failure becomes: its HTTP status, and the sentence the person reads."""

    status: int
    detail: str


def superset_refusal(exc: BaseException) -> Refused | None:
    """The response `exc` becomes on a /v1 route, or None when it is not Superset's or Superset mode's to answer.

    The order matters: each class is read before any class it derives from."""
    if isinstance(exc, SupersetRefusedCaller):
        return Refused(401, SIGN_IN_AGAIN)
    if isinstance(exc, (NotABuilder, SupersetPermissionDenied)):
        return Refused(403, NOT_A_BUILDER)
    if isinstance(exc, SupersetAccessRefused):
        return Refused(403, ACCESS_REFUSED)
    if isinstance(exc, SupersetRefusal):  # `SupersetDenied`, and any refusal not read above
        return Refused(403, DENIED)
    if isinstance(exc, SupersetMisconfigured):
        return Refused(503, exc.message)
    if isinstance(exc, (SupersetToolError, SupersetUnavailable)):  # a response too large to send among them
        return Refused(503, str(exc))
    if isinstance(exc, SupersetCredentialMissing):
        return Refused(401, str(exc))
    if isinstance(exc, SupersetReadOnly):
        return Refused(403, str(exc))
    return None
