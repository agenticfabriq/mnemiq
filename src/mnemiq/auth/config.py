"""Every rule a deployment's identity settings must meet, checked once at startup.

A door that cannot enforce its configuration should not open. Each refusal names the setting to change and never
repeats a secret's value, because this message reaches the console and whatever log the operator keeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

OPENWEBUI_ISSUER = "open-webui"  # the fixed `iss` Open WebUI puts on its signed user header
MIN_SECRET_BYTES = 32  # an HS256 token can be attacked offline, so a short secret can be guessed and any user minted
_LOOPBACK_LITERALS = frozenset({"127.0.0.1", "::1"})


class AuthConfigError(SystemExit):
    """A configuration the doors cannot enforce. A SystemExit, so the server stops with the message alone."""


@dataclass(frozen=True)
class OidcSettings:
    issuer: str
    audience: str  # /v1 and MCP
    openwebui_audience: str | None  # /openai/v1, for option A
    jwks_url: str | None
    jwks_file: str | None
    tenant_claim: str
    roles_claim: str
    groups_claim: str | None  # the opt-in: groups pass through only when this is set
    leeway_s: float


@dataclass(frozen=True)
class OpenWebUISettings:
    secret: bytes
    header: str  # lowercase
    max_lifetime_s: int
    leeway_s: float


@dataclass(frozen=True)
class AuthConfig:
    tenant: str  # "*" accepts any tenant
    oidc: OidcSettings | None
    openwebui: OpenWebUISettings | None
    binding: tuple[str, bytes] | None  # the connection key's header (lowercase) and value
    user_headers: tuple[str, ...]  # Open WebUI's unsigned user headers (lowercase), refused on every door
    map_path: str | None
    mcp_http: bool
    # The signed header's name (lowercase), known even with that verifier off, so /v1 can refuse it by name.
    jwt_header: str = "x-openwebui-user-jwt"

    @property
    def issuers(self) -> set[str]:
        """The issuers a token may come from here, which are the only keys the identity map may use."""
        found = set()
        if self.oidc is not None:
            found.add(self.oidc.issuer)
        if self.openwebui is not None:
            found.add(OPENWEBUI_ISSUER)
        return found


def check_mcp_audience(audience: str) -> None:
    """Under MCP over HTTP the audience is the MCP door's own URL, compared exactly.

    mnemiq's `aud` check is an exact string match, and an identity provider's audience mapper adds the configured
    string verbatim, so a near variant (a trailing slash, a query) would pass a client's looser check and then fail
    every request as a 401. Refusing it here turns that into a startup error. https only, except the loopback IP
    literals for local development: `localhost` is a name, and a name can resolve elsewhere.
    """
    problem = None
    try:
        parts = urlsplit(audience)
        port_ok = parts.port is None or parts.port > 0
    except ValueError:
        parts, port_ok = None, False
    if parts is None or not port_ok:
        problem = "it is not a URL with a valid port"
    elif parts.query or parts.fragment or audience.endswith(("?", "#")):
        problem = "it has a query or a fragment"
    elif "@" in parts.netloc:
        problem = "it carries a user name"
    elif not parts.hostname:
        problem = "it names no host"
    elif parts.path != "/mcp":
        problem = "its path is not exactly /mcp"
    elif parts.scheme == "http" and parts.hostname not in _LOOPBACK_LITERALS:
        problem = "http is allowed only for 127.0.0.1 or [::1]"
    elif parts.scheme not in ("https", "http"):
        problem = "its scheme is not https"
    if problem:
        raise AuthConfigError(
            f"MNEMIQ_OIDC_AUDIENCE must be the MCP door's own URL, https://<host>/mcp, when MNEMIQ_MCP_TRANSPORT=http: "
            f"{problem} (got {audience!r})")


def _blank(value: str | None) -> bool:
    return value is None or not value.strip()


def auth_config_from_settings(settings, *, validate_map=None) -> AuthConfig | None:
    """The deployment's identity configuration, or None when no door verifies tokens (the settings identity serves).

    `validate_map(path, *, issuers, groups_passthrough, oidc_issuer)` checks the identity map's startup rules; it
    defaults to the identity map's own check.
    """
    oidc_on = not _blank(settings.oidc_issuer)
    # Blank is unset: a sourced copy of .env.example sets every unset setting to the empty string.
    openwebui_on = not _blank(settings.openwebui_jwt_secret)
    mcp_http = settings.mcp_transport == "http"
    binding_header, binding_key = settings.openwebui_connection_key_header, settings.openwebui_connection_key

    if not oidc_on and not openwebui_on:
        if mcp_http:
            raise AuthConfigError("MNEMIQ_MCP_TRANSPORT=http needs MNEMIQ_OIDC_ISSUER: MCP over HTTP serves each "
                                  "caller as their own token says, and with no issuer there is none to verify")
        if settings.identity_map_path:
            raise AuthConfigError("MNEMIQ_IDENTITY_MAP_PATH is set but no token verifier is configured: set "
                                  "MNEMIQ_OIDC_ISSUER or MNEMIQ_OPENWEBUI_JWT_SECRET, or unset the map")
        if settings.openwebui_audience:
            raise AuthConfigError("MNEMIQ_OPENWEBUI_AUDIENCE needs MNEMIQ_OIDC_ISSUER, whose tokens it is the "
                                  "audience of")
        if binding_header or binding_key:
            raise AuthConfigError("MNEMIQ_OPENWEBUI_CONNECTION_KEY is set but no Open WebUI door is configured: set "
                                  "MNEMIQ_OPENWEBUI_JWT_SECRET or MNEMIQ_OPENWEBUI_AUDIENCE, or unset the key")
        return None

    if _blank(settings.tenant):
        raise AuthConfigError("MNEMIQ_TENANT is required with a token verifier: name the tenant every token must "
                              "carry, or `*` for a deployment that accepts any tenant")
    tenant = settings.tenant.strip()

    oidc = None
    if oidc_on:
        if _blank(settings.oidc_audience):
            raise AuthConfigError("MNEMIQ_OIDC_AUDIENCE is required with MNEMIQ_OIDC_ISSUER: the `aud` a token "
                                  "for this deployment carries, so a token minted for another service is refused")
        if settings.oidc_jwks_url and settings.oidc_jwks_file:
            raise AuthConfigError("set MNEMIQ_OIDC_JWKS_URL or MNEMIQ_OIDC_JWKS_FILE, not both")
        if settings.openwebui_audience is not None and settings.openwebui_audience == settings.oidc_audience:
            raise AuthConfigError("MNEMIQ_OPENWEBUI_AUDIENCE must differ from MNEMIQ_OIDC_AUDIENCE: each door "
                                  "refuses a token carrying the other's audience, so equal values refuse every token")
        if mcp_http:
            check_mcp_audience(settings.oidc_audience)
        oidc = OidcSettings(
            issuer=settings.oidc_issuer, audience=settings.oidc_audience,
            openwebui_audience=settings.openwebui_audience or None,
            jwks_url=settings.oidc_jwks_url, jwks_file=settings.oidc_jwks_file,
            tenant_claim=settings.oidc_tenant_claim, roles_claim=settings.oidc_roles_claim,
            groups_claim=settings.oidc_groups_claim or None, leeway_s=settings.oidc_leeway_s)
    else:
        if mcp_http:
            raise AuthConfigError("MNEMIQ_MCP_TRANSPORT=http needs MNEMIQ_OIDC_ISSUER: the MCP door accepts an "
                                  "identity provider's token only")
        if settings.openwebui_audience:
            raise AuthConfigError("MNEMIQ_OPENWEBUI_AUDIENCE needs MNEMIQ_OIDC_ISSUER, whose tokens it is the "
                                  "audience of")

    openwebui = None
    if openwebui_on:
        secret = settings.openwebui_jwt_secret.encode()
        if len(secret) < MIN_SECRET_BYTES:
            raise AuthConfigError(f"MNEMIQ_OPENWEBUI_JWT_SECRET must be at least {MIN_SECRET_BYTES} bytes")
        if tenant == "*":
            raise AuthConfigError("Open WebUI's signed header carries no tenant, so MNEMIQ_TENANT names one tenant "
                                  "for its users, not `*`")
        if not binding_header or not binding_key:
            raise AuthConfigError("Open WebUI's signed header needs MNEMIQ_OPENWEBUI_CONNECTION_KEY_HEADER and "
                                  "MNEMIQ_OPENWEBUI_CONNECTION_KEY: Open WebUI sends the same token to every backend "
                                  "it calls, and the key is what shows a request came through mnemiq's connection")
        openwebui = OpenWebUISettings(secret=secret, header=settings.openwebui_jwt_header.lower(),
                                      max_lifetime_s=settings.openwebui_jwt_max_lifetime_s,
                                      leeway_s=settings.oidc_leeway_s)

    user_headers = tuple(h.strip().lower() for h in settings.openwebui_user_headers.split(",") if h.strip())
    binding = None
    if binding_header or binding_key:
        if not openwebui_on and not settings.openwebui_audience:
            raise AuthConfigError("MNEMIQ_OPENWEBUI_CONNECTION_KEY is set but no Open WebUI door is configured: set "
                                  "MNEMIQ_OPENWEBUI_JWT_SECRET or MNEMIQ_OPENWEBUI_AUDIENCE, or unset the key")
        if not binding_header or not binding_key:
            raise AuthConfigError("set both MNEMIQ_OPENWEBUI_CONNECTION_KEY_HEADER and "
                                  "MNEMIQ_OPENWEBUI_CONNECTION_KEY, or neither")
        if len(binding_key.encode()) < MIN_SECRET_BYTES:
            raise AuthConfigError(f"MNEMIQ_OPENWEBUI_CONNECTION_KEY must be at least {MIN_SECRET_BYTES} bytes")
        # Every header a door reads as identity: the key binds the connection, so it may never alias one.
        identity_headers = {"authorization", settings.openwebui_jwt_header.strip().lower(), *user_headers}
        if binding_header.strip().lower() in identity_headers:
            raise AuthConfigError("MNEMIQ_OPENWEBUI_CONNECTION_KEY_HEADER must be a header of its own, not "
                                  "Authorization, the signed user header or one of Open WebUI's plain user headers: "
                                  "the key binds the connection, it is not a credential")
        binding = (binding_header.strip().lower(), binding_key.encode())

    config = AuthConfig(
        tenant=tenant, oidc=oidc, openwebui=openwebui, binding=binding,
        user_headers=user_headers, map_path=settings.identity_map_path or None, mcp_http=mcp_http,
        jwt_header=settings.openwebui_jwt_header.strip().lower())
    if config.map_path:
        if validate_map is None:
            from mnemiq.auth.identity_map import IdentityMap

            validate_map = IdentityMap.validate_for_startup
        validate_map(config.map_path, issuers=config.issuers, groups_passthrough=bool(oidc and oidc.groups_claim),
                     oidc_issuer=oidc.issuer if oidc else None)
    return config
