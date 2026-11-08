"""The identity map: roles for a token that carries none of its own, written by the operator, keyed by issuer.

    {"<issuer>": {"sub": {"<subject>": ["role", ...]},
                  "groups": {"<group>": ["role", ...]},
                  "email": {"<address>": ["role", ...]}}}

A rule written for one issuer never matches another issuer's subject or group of the same name. It is read from
disk on every request, as the access policy is, with no cached copy: an edit takes effect at once, and a broken
edit denies at once rather than leaving the last good roles in force. Any file it cannot read or accept denies
every identity, loudly. It is deliberately static: managing identities across sources is the control plane's job.
"""

from __future__ import annotations

import json
import logging

from mnemiq.auth.config import OPENWEBUI_ISSUER, AuthConfigError
from mnemiq.auth.roles import normalize_roles
from mnemiq.auth.token import TokenRejected

logger = logging.getLogger("mnemiq.auth")

CLAIMS = ("sub", "groups", "email")


class IdentityMapUnavailable(TokenRejected):
    """The map could not be read or accepted. Every identity that needs it is denied; the doors answer 503."""

    def __init__(self) -> None:
        super().__init__("identity map unavailable")


def fold_email(address: str) -> str:
    """ASCII letters lowercased, everything else kept as written. Unicode case folding would merge addresses that are
    different mailboxes: `ß` folds to `ss` (`faß.de` and `fass.de` are separately registrable domains) and the
    Kelvin sign to `k`."""
    return "".join(c.lower() if c.isascii() else c for c in address)


class _Conflict(ValueError):
    """A rule that contradicts the deployment's configuration, which refuses to start rather than deny."""


class IdentityMap:
    def __init__(self, rules: dict[str, dict[str, dict[str, list[str]]]]) -> None:
        self._rules = rules

    @classmethod
    def read(cls, path: str, *, issuers: set[str], groups_passthrough: bool = False,
             oidc_issuer: str | None = None) -> IdentityMap:
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("identity map %s could not be read (%s); denying every identity", path, type(exc).__name__)
            raise IdentityMapUnavailable() from exc
        try:
            return cls(cls._parse(data, issuers, groups_passthrough, oidc_issuer))
        except ValueError as exc:
            # The message names issuers and claim keys only, never a subject, group or address from the file.
            logger.error("identity map %s is invalid (%s); denying every identity", path, exc)
            raise IdentityMapUnavailable() from exc

    @classmethod
    def validate_for_startup(cls, path: str, *, issuers: set[str], groups_passthrough: bool,
                             oidc_issuer: str | None) -> None:
        """Refuse to start on a rule that contradicts the configuration. A file that cannot be read or parsed is
        logged and left to deny at request time, the access policy's rule, so fixing it needs no restart."""
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("identity map %s could not be read (%s); every identity will be denied until it can",
                         path, type(exc).__name__)
            return
        try:
            cls._parse(data, issuers, groups_passthrough, oidc_issuer)
        except _Conflict as exc:
            raise AuthConfigError(f"MNEMIQ_IDENTITY_MAP_PATH: {exc}") from exc
        except ValueError as exc:
            logger.error("identity map %s is invalid (%s); every identity will be denied until it is fixed", path,
                         exc)

    @staticmethod
    def _parse(data, issuers, groups_passthrough, oidc_issuer) -> dict[str, dict[str, dict[str, list[str]]]]:
        if not isinstance(data, dict):
            raise ValueError("the map is not an object keyed by issuer")
        rules: dict[str, dict[str, dict[str, list[str]]]] = {}
        for issuer, by_claim in data.items():
            if issuer not in issuers:
                raise ValueError(f"issuer {issuer!r} is not one this deployment verifies")
            if not isinstance(by_claim, dict):
                raise ValueError(f"the rules under {issuer!r} are not an object keyed by claim")
            for claim, by_value in by_claim.items():
                if claim not in CLAIMS:
                    raise ValueError(f"unknown claim {claim!r} under {issuer!r}; the claims are {list(CLAIMS)}")
                if issuer == OPENWEBUI_ISSUER and claim != "sub":
                    raise _Conflict(f"open-webui accepts sub rules only, and this map has {claim!r} rules there: "
                                    "Open WebUI's email is unverified, and it sends no groups")
                if claim == "groups" and groups_passthrough and issuer == oidc_issuer:
                    raise _Conflict("groups rules for the issuer are refused while MNEMIQ_OIDC_GROUPS_CLAIM passes "
                                    "its groups through: the two would contradict each other")
                if not isinstance(by_value, dict):
                    raise ValueError(f"the {claim!r} rules under {issuer!r} are not an object")
                for value, roles in by_value.items():
                    if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
                        raise ValueError(f"a {claim!r} rule under {issuer!r} maps to something other than a list "
                                         "of role names")
                    key = fold_email(value) if claim == "email" else value
                    rules.setdefault(issuer, {}).setdefault(claim, {}).setdefault(key, []).extend(
                        normalize_roles(roles))
        return rules

    def roles_for(self, issuer: str, *, sub: str, email: str | None, groups: list[str],
                  roles_claim_present: bool) -> list[str]:
        """The roles every matching rule of this issuer names, unioned and sorted. `email` is passed only when it is
        proven. `groups` rules apply to every token; `sub` and `email` rules only to a token without a roles claim,
        because a present claim, even an empty one, is the provider's own answer."""
        rules = self._rules.get(issuer, {})
        found: set[str] = set()
        if not roles_claim_present:
            found.update(rules.get("sub", {}).get(sub, ()))
            if email is not None:
                found.update(rules.get("email", {}).get(fold_email(email), ()))
        for group in groups:
            found.update(rules.get("groups", {}).get(group, ()))
        return sorted(found)

    def has_subject_rules(self, issuer: str) -> bool:
        rules = self._rules.get(issuer, {})
        return bool(rules.get("sub") or rules.get("email"))

    def warn_undeclared_roles(self, authz) -> None:
        """At startup: a mapped role the access policy does not declare grants nothing, which is a typo more often
        than an intent."""
        declared = getattr(authz, "policy_roles", None)
        if declared is None:
            return  # a provider that cannot enumerate roles cannot be checked
        known = set(declared())
        mapped = {role for by_claim in self._rules.values() for by_value in by_claim.values()
                  for roles in by_value.values() for role in roles}
        for role in sorted(mapped - known):
            logger.warning("identity map names role %r, which the access policy does not declare: it grants nothing",
                           role)
