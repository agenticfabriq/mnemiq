"""A verified token's claims, turned into the IdentityContext the engine enforces.

The token says who; the access policy still says what. Roles come from the token's roles claim when it has one,
and from the identity map when it has none. A token's groups grant nothing unless the operator maps them or opts
the provider in (`MNEMIQ_OIDC_GROUPS_CLAIM`), because many providers let users create groups, and a group named
like a policy role would otherwise grant that role. The tenant is always checked here, at the door, so the
policy needs no tenant check of its own.
"""

from __future__ import annotations

import logging
import threading

from mnemiq.auth.config import OPENWEBUI_ISSUER, AuthConfig
from mnemiq.auth.identity_map import IdentityMap
from mnemiq.auth.roles import normalize_roles
from mnemiq.auth.token import TokenRejected
from mnemiq.contract import IdentityContext

logger = logging.getLogger("mnemiq.auth")

_SKIP_LOGGED: set[str] = set()  # issuers already told that their sub and email rules wait for an absent roles claim
_SKIP_LOCK = threading.Lock()
_WRONG_TYPE = object()


class TenantMismatch(TokenRejected):
    """A valid token for another tenant: a 403 on the HTTP API."""


def _claim_path(found: dict, path: str):
    """(present, value) for a dotted path. A step through something that is not an object is present with the
    wrong type, never absent: absent would hand a token that did send roles to the map's subject rules."""
    node = found
    for part in path.split("."):
        if not isinstance(node, dict):
            return True, _WRONG_TYPE
        if part not in node:
            return False, None
        node = node[part]
    return True, node


def _strings(value, what: str, sub: str, jti: str | None) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TokenRejected(f"{what} claim has the wrong type", sub=sub, jti=jti)
    return value


def _read_map(cfg: AuthConfig) -> IdentityMap | None:
    if not cfg.map_path:
        return None
    oidc = cfg.oidc
    return IdentityMap.read(cfg.map_path, issuers=cfg.issuers, groups_passthrough=bool(oidc and oidc.groups_claim),
                            oidc_issuer=oidc.issuer if oidc else None)


def identity_from_oidc(found: dict, cfg: AuthConfig) -> IdentityContext:
    oidc = cfg.oidc
    sub = str(found["sub"])
    jti = found.get("jti") if isinstance(found.get("jti"), str) else None

    tenant = found.get(oidc.tenant_claim)
    if not isinstance(tenant, str) or not tenant.strip():
        raise TokenRejected("token carries no tenant", sub=sub, jti=jti)
    if cfg.tenant != "*" and tenant != cfg.tenant:
        raise TenantMismatch("token is for another tenant", sub=sub, jti=jti)

    present, raw_roles = _claim_path(found, oidc.roles_claim)
    roles = normalize_roles(_strings(raw_roles, "roles", sub, jti)) if present else []

    # The provider's own groups, read for the map's groups rules; they reach the identity only through the opt-in.
    token_groups = _strings(found["groups"], "groups", sub, jti) if "groups" in found else []
    groups: list[str] = []
    if oidc.groups_claim:
        groups = normalize_roles(_strings(found.get(oidc.groups_claim, []), "groups", sub, jti))

    email = found.get("email") if found.get("email_verified") is True and isinstance(found.get("email"), str) else None

    identity_map = _read_map(cfg)
    if identity_map is not None:
        mapped = identity_map.roles_for(oidc.issuer, sub=sub, email=email, groups=token_groups,
                                        roles_claim_present=present)
        roles = list(dict.fromkeys([*roles, *mapped]))
        if present and identity_map.has_subject_rules(oidc.issuer):
            with _SKIP_LOCK:
                first = oidc.issuer not in _SKIP_LOGGED
                _SKIP_LOGGED.add(oidc.issuer)
            if first:
                logger.info("tokens from %s carry a roles claim, so the identity map's sub and email rules for it are "
                            "skipped; assign those roles at the identity provider", oidc.issuer)
    return IdentityContext(tenant_id=tenant, principal_id=sub, email=email, roles=roles, groups=groups)


def identity_from_openwebui(found: dict, cfg: AuthConfig) -> IdentityContext:
    """Open WebUI's token carries no tenant, no groups and no data roles: its tenant is the deployment's, its roles
    come from `sub` rules only, and its unverified email goes nowhere. The principal names the issuer, so an audit
    record never confuses an Open WebUI user id with a provider's subject of the same value."""
    sub = str(found["sub"])
    identity_map = _read_map(cfg)
    roles = (identity_map.roles_for(OPENWEBUI_ISSUER, sub=sub, email=None, groups=[], roles_claim_present=False)
             if identity_map is not None else [])
    return IdentityContext(tenant_id=cfg.tenant, principal_id=f"{OPENWEBUI_ISSUER}:{sub}", roles=roles)
