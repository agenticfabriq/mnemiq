"""The verifier against a real Keycloak: real tokens, real JWKS, and a key rotation. Opt-in, never in CI.

Run Keycloak on loopback with the test realm imported, then point the test at it:

    docker run --rm -d --name mnemiq-kc -p 127.0.0.1:18089:8080 \\
      -e KC_BOOTSTRAP_ADMIN_USERNAME=admin -e KC_BOOTSTRAP_ADMIN_PASSWORD="$KC_ADMIN_PASSWORD" \\
      -v "$PWD/tests/fixtures/keycloak:/opt/keycloak/data/import:ro" \\
      quay.io/keycloak/keycloak:26.0 start-dev --import-realm
    MNEMIQ_LIVE_KEYCLOAK_URL=http://127.0.0.1:18089 MNEMIQ_LIVE_KEYCLOAK_ADMIN_PASSWORD="$KC_ADMIN_PASSWORD" \\
      uv run pytest tests/test_live_keycloak.py -v

The realm's users and their test-only passwords are in tests/fixtures/keycloak/mnemiq-realm.json.
"""

import json
import os
from pathlib import Path

import pytest

URL = os.environ.get("MNEMIQ_LIVE_KEYCLOAK_URL", "").rstrip("/")
pytestmark = pytest.mark.skipif(not URL, reason="set MNEMIQ_LIVE_KEYCLOAK_URL to run against a live Keycloak")

REALM_FILE = Path(__file__).parent / "fixtures" / "keycloak" / "mnemiq-realm.json"
REALM = json.loads(REALM_FILE.read_text())
ISSUER = f"{URL}/realms/{REALM['realm']}"
PASSWORDS = {u["username"]: u["credentials"][0]["value"] for u in REALM["users"]}
TABLES = {"analyst": ["claim"], "store1_manager": ["store"]}


def _token(user: str, client: str = "mnemiq-cli") -> str:
    import httpx

    r = httpx.post(f"{ISSUER}/protocol/openid-connect/token",
                   data={"grant_type": "password", "client_id": client, "username": user,
                         "password": PASSWORDS[user], "scope": "openid"}, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def _settings(**over):
    from mnemiq.config import Settings

    base = dict(tenant="acme", oidc_issuer=ISSUER, oidc_audience="mnemiq-api", openwebui_audience="mnemiq-openwebui")
    base.update(over)
    return Settings(**base)


class _RT:
    def scope(self, identity):
        return {"tables": sorted(t for role in identity.roles for t in TABLES.get(role, [])), "starters": []}


def _client():
    from fastapi.testclient import TestClient

    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.doors import HttpDoors
    from mnemiq.contract import IdentityContext
    from mnemiq.server.app import build_app

    doors = HttpDoors(auth_config_from_settings(_settings()))
    return TestClient(build_app(_RT(), IdentityContext(tenant_id="local", principal_id="local"), doors=doors))


def test_two_real_users_see_two_scopes():
    client = _client()
    ana = client.get("/v1/schema", headers={"Authorization": f"Bearer {_token('ana')}"})
    sam = client.get("/v1/schema", headers={"Authorization": f"Bearer {_token('sam')}"})
    assert ana.status_code == 200, ana.text
    assert ana.json()["tables"] == ["claim"] and sam.json()["tables"] == ["store"]
    who = client.get("/v1/whoami", headers={"Authorization": f"Bearer {_token('ana')}"}).json()
    assert who["tenant_id"] == "acme" and who["principal_id"]


def test_a_token_for_the_open_webui_path_is_refused_on_v1():
    r = _client().get("/v1/schema", headers={"Authorization": f"Bearer {_token('ana', 'mnemiq-openwebui-cli')}"})
    assert r.status_code == 401 and "audience" in r.json()["detail"]


def test_no_token_is_refused():
    assert _client().get("/v1/schema").status_code == 401


def test_a_real_token_without_db_read_is_a_403():
    """The client signing users in to the workbench must request db.read; one that does not gets a 403."""
    import jwt

    token = _token("ana", "mnemiq-no-scope-cli")
    assert "db.read" not in jwt.decode(token, options={"verify_signature": False}).get("scope", "").split()
    r = _client().get("/v1/schema", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403 and "db.read" in r.json()["detail"]


def test_a_rotated_key_verifies_after_one_refetch():
    """Keycloak signs with a new key; the verifier, which already holds the old set, refetches once its interval has
    passed and verifies the new token. Within the interval, the new `kid` is refused without a fetch."""
    import httpx
    import jwt

    from mnemiq.auth.config import auth_config_from_settings
    from mnemiq.auth.token import JwtIdentityVerifier, TokenRejected

    admin_password = os.environ.get("MNEMIQ_LIVE_KEYCLOAK_ADMIN_PASSWORD")
    if not admin_password:
        pytest.skip("set MNEMIQ_LIVE_KEYCLOAK_ADMIN_PASSWORD to rotate the realm's key")

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    oidc = auth_config_from_settings(_settings()).oidc
    verifier = JwtIdentityVerifier(oidc, audience="mnemiq-api", refuse_audience="mnemiq-openwebui", clock=clock)
    old = _token("ana")
    ana = verifier.claims(old)["sub"]

    admin = httpx.post(f"{URL}/realms/master/protocol/openid-connect/token",
                       data={"grant_type": "password", "client_id": "admin-cli", "username": "admin",
                             "password": admin_password}, timeout=10).json()["access_token"]
    realm_id = httpx.get(f"{URL}/admin/realms/{REALM['realm']}", headers={"Authorization": f"Bearer {admin}"},
                         timeout=10).json()["id"]
    created = httpx.post(f"{URL}/admin/realms/{REALM['realm']}/components",
                         headers={"Authorization": f"Bearer {admin}"}, timeout=10,
                         json={"name": "rotated", "providerId": "rsa-generated",
                               "providerType": "org.keycloak.keys.KeyProvider", "parentId": realm_id,
                               "config": {"priority": ["500"], "keySize": ["2048"], "algorithm": ["RS256"]}})
    assert created.status_code == 201, created.text

    new = _token("ana")
    assert jwt.get_unverified_header(new)["kid"] != jwt.get_unverified_header(old)["kid"]
    with pytest.raises(TokenRejected, match="unknown signing key"):
        verifier.claims(new)  # within the refetch interval: refused, no fetch
    clock.now += 61
    assert verifier.claims(new)["sub"] == ana
    assert verifier.claims(old)["sub"] == ana  # the old key is still published, still valid
