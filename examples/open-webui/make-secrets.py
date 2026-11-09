#!/usr/bin/env python3
"""The example's secrets, users and settings, in three steps run from the repository root:

    python examples/open-webui/make-secrets.py secrets   # before `docker compose up`: every secret, kept on re-runs
    python examples/open-webui/make-secrets.py users     # after it: Keycloak's client secrets and demo users, and
                                                         # Open WebUI's groups and models (and option B's, if running)
    python examples/open-webui/make-secrets.py check     # each service receives every secret it needs

Secrets live in examples/open-webui/.secrets/ (git-ignored): one state file and the .env files Compose reads, all
mode 600. Nothing here prints a secret. The demo users' passwords are in .secrets/users.txt for you to read.
Standard library only, so it runs with any Python 3.10+.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECRETS = HERE / ".secrets"
STATE = SECRETS / "state.json"
COMPOSE = ["docker", "compose", "-f", str(HERE / "compose.yaml")]
KEYCLOAK = "http://keycloak.localhost:18089"
REALM = "mnemiq"
OPEN_WEBUI = {"a": "http://localhost:13000", "b": "http://localhost:13001"}
PROFILES = {"frontier": ("analyst", "store1_manager"), "for-analysts": ("analyst",), "for-stores": ("store1_manager",)}
KEYCLOAK_USERS = {"ana": "analyst", "sam": "store1_manager"}
OPTION_B_USERS = {"bea": "analyst", "bob": "store1_manager"}
LLM_SETTINGS = ("MNEMIQ_LLM_BASE_URL", "MNEMIQ_LLM_API_KEY", "MNEMIQ_LLM_MODEL", "MNEMIQ_EMBED_MODEL",
                "MNEMIQ_EMBED_BASE_URL", "MNEMIQ_EMBED_API_KEY")
# Everything Open WebUI could add to a mnemiq model is off: its built-in tools would arrive as `tools`, which the
# endpoint refuses, and the rest would send the model's chats elsewhere.
CAPABILITIES = {"builtin_tools": False, "vision": False, "file_upload": False, "web_search": False,
                "code_interpreter": False, "image_generation": False}
# What `check` requires each service to receive, by name.
NEEDED = {
    "keycloak": ["KC_BOOTSTRAP_ADMIN_PASSWORD"],
    "oauth2-proxy": ["OAUTH2_PROXY_COOKIE_SECRET", "PROXY_CLIENT_SECRET"],
    "init": ["MNEMIQ_LLM_BASE_URL", "MNEMIQ_LLM_MODEL"],
    "mnemiq": ["MNEMIQ_OPENWEBUI_JWT_SECRET", "MNEMIQ_OPENWEBUI_CONNECTION_KEY", "MNEMIQ_LLM_BASE_URL",
               "MNEMIQ_LLM_MODEL"],
    "open-webui": ["WEBUI_SECRET_KEY", "OAUTH_CLIENT_SECRET", "OPENAI_API_CONFIGS"],
    "open-webui-b": ["WEBUI_SECRET_KEY", "FORWARD_USER_INFO_HEADER_JWT_SECRET", "OPENAI_API_CONFIGS"],
}


def write_private(path: Path, text: str, mode: int = 0o600) -> None:
    path.write_text(text)
    path.chmod(mode)


def env_file(values: dict[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in values.items())


def connection(state: dict, option: str) -> str:
    """Open WebUI's connection to mnemiq, as OPENAI_API_CONFIGS: the {{TASK}} header on every request, the connection
    key, and the fixed model list, so Open WebUI never asks mnemiq for models and the group grants are the only list."""
    config = {"enable": True, "auth_type": "system_oauth" if option == "a" else "none",
              "headers": {"X-Mnemiq-Task": "{{TASK}}", "X-Mnemiq-Connection": state["connection_key"]},
              "model_ids": list(PROFILES)}
    return json.dumps({"0": config}, separators=(",", ":"))


def cmd_secrets() -> int:
    SECRETS.mkdir(mode=0o700, exist_ok=True)
    SECRETS.chmod(0o700)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for name in ("kc_admin", "proxy_client", "openwebui_client", "webui_key_a", "webui_key_b", "jwt", "connection_key",
                 "admin_a", "admin_b", *KEYCLOAK_USERS, *OPTION_B_USERS):
        state.setdefault(name, secrets.token_urlsafe(32))
    state.setdefault("cookie", secrets.token_hex(16))  # oauth2-proxy wants 16, 24 or 32 bytes
    write_private(STATE, json.dumps(state, indent=1) + "\n")
    write_private(SECRETS / "keycloak.env", env_file(
        {"KC_BOOTSTRAP_ADMIN_USERNAME": "admin", "KC_BOOTSTRAP_ADMIN_PASSWORD": state["kc_admin"]}))
    write_private(SECRETS / "oauth2-proxy.env", env_file(
        {"OAUTH2_PROXY_COOKIE_SECRET": state["cookie"], "PROXY_CLIENT_SECRET": state["proxy_client"]}))
    write_private(SECRETS / "mnemiq.env", env_file(
        {"MNEMIQ_OPENWEBUI_JWT_SECRET": state["jwt"], "MNEMIQ_OPENWEBUI_CONNECTION_KEY": state["connection_key"]}))
    write_private(SECRETS / "open-webui.env", env_file(
        {"WEBUI_SECRET_KEY": state["webui_key_a"], "OAUTH_CLIENT_SECRET": state["openwebui_client"],
         "OPENAI_API_CONFIGS": connection(state, "a")}))
    write_private(SECRETS / "open-webui-b.env", env_file(
        {"WEBUI_SECRET_KEY": state["webui_key_b"], "FORWARD_USER_INFO_HEADER_JWT_SECRET": state["jwt"],
         "OPENAI_API_CONFIGS": connection(state, "b")}))
    users = [f"Keycloak (workbench and Open WebUI option A, {KEYCLOAK}): {u} / {state[u]}" for u in KEYCLOAK_USERS]
    users += [f"Open WebUI option A admin ({OPEN_WEBUI['a']}): admin-a@example.test / {state['admin_a']}",
              f"Open WebUI option B admin ({OPEN_WEBUI['b']}): admin-b@example.test / {state['admin_b']}"]
    users += [f"Open WebUI option B ({OPEN_WEBUI['b']}): {u}@example.test / {state[u]}" for u in OPTION_B_USERS]
    write_private(SECRETS / "users.txt", "\n".join(users) + "\n")
    identity_map = SECRETS / "identity-map.json"
    if not identity_map.exists():
        # Not a secret: Open WebUI user ids for option B, filled by `users`. Readable by mnemiq's container user.
        write_private(identity_map, json.dumps({"open-webui": {"sub": {}}}, indent=1) + "\n", 0o644)
    llm = SECRETS / "llm.env"
    found = {key: os.environ[key] for key in LLM_SETTINGS if os.environ.get(key)}
    if found:
        write_private(llm, env_file(found))
    elif not llm.exists():
        write_private(llm, env_file({key: "" for key in LLM_SETTINGS}))
    missing = [k for k in ("MNEMIQ_LLM_BASE_URL", "MNEMIQ_LLM_MODEL") if not _env_value(llm, k)]
    print(f"wrote the secrets to {SECRETS} (mode 600); demo passwords are in {SECRETS / 'users.txt'}")
    if missing:
        print(f"set {', '.join(missing)} (and the key) in {llm}, or export them and run this again: mnemiq needs "
              "its model to enrich the demo data and to answer")
        return 2
    return 0


def _env_value(path: Path, key: str) -> str:
    for line in path.read_text().splitlines():
        name, _, value = line.partition("=")
        if name == key:
            return value
    return ""


def request(method: str, url: str, *, token: str | None = None, data=None, form: dict | None = None,
            ok=(200, 201, 204)):
    headers = {"Accept": "application/json"}
    body = None
    if form is not None:
        body = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as exc:
        if exc.code in ok:
            return exc.code, None
        return exc.code, exc.read().decode(errors="replace")[:300]


def wait_for(url: str, what: str, seconds: int = 120) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(3)
    print(f"{what} did not answer at {url} within {seconds} s: start the stack, then run `users` again")
    return False


def keycloak(state: dict) -> None:
    status, tok = request("POST", f"{KEYCLOAK}/realms/master/protocol/openid-connect/token",
                          form={"grant_type": "password", "client_id": "admin-cli", "username": "admin",
                                "password": state["kc_admin"]})
    if status != 200:
        raise SystemExit(f"Keycloak refused the admin sign-in ({status})")
    admin = tok["access_token"]
    base = f"{KEYCLOAK}/admin/realms/{REALM}"
    for client_id, key in (("workbench-proxy", "proxy_client"), ("open-webui", "openwebui_client")):
        _, found = request("GET", f"{base}/clients?clientId={client_id}", token=admin)
        client = found[0]
        client["secret"] = state[key]
        status, _ = request("PUT", f"{base}/clients/{client['id']}", token=admin, data=client)
        print(f"Keycloak: {client_id}'s secret set ({status})")
    for user, role in KEYCLOAK_USERS.items():
        rep = {"username": user, "email": f"{user}@example.test", "emailVerified": True, "enabled": True,
               "firstName": user.title(), "lastName": "Example", "groups": [f"/{role}"],
               "credentials": [{"type": "password", "value": state[user], "temporary": False}]}
        status, _ = request("POST", f"{base}/users", token=admin, data=rep, ok=(201, 409))
        _, found = request("GET", f"{base}/users?username={user}&exact=true", token=admin)
        uid = found[0]["id"]
        request("PUT", f"{base}/users/{uid}/reset-password", token=admin,
                data={"type": "password", "value": state[user], "temporary": False})
        _, role_rep = request("GET", f"{base}/roles/{role}", token=admin)
        request("POST", f"{base}/users/{uid}/role-mappings/realm", token=admin, data=[role_rep])
        print(f"Keycloak: {user} ({role}) {'created' if status == 201 else 'already there, password reset'}")


def open_webui(state: dict, option: str) -> dict[str, str]:
    """Groups that mirror the profiles' roles, and one model per profile with every extra off and read granted to
    the groups holding its roles. Returns option B's user ids by role."""
    base = OPEN_WEBUI[option]
    email, password = f"admin-{option}@example.test", state[f"admin_{option}"]
    status, body = request("POST", f"{base}/api/v1/auths/signup",
                           data={"name": f"Admin {option.upper()}", "email": email, "password": password})
    if status != 200:
        status, body = request("POST", f"{base}/api/v1/auths/signin", data={"email": email, "password": password})
    if status != 200:
        raise SystemExit(f"Open WebUI {option.upper()} refused its admin ({status})")
    admin = body["token"]
    _, listed = request("GET", f"{base}/api/v1/groups/", token=admin)
    groups = {g["name"]: g["id"] for g in (listed or [])}
    for role in {r for roles in PROFILES.values() for r in roles}:
        if role not in groups:
            _, made = request("POST", f"{base}/api/v1/groups/create", token=admin,
                              data={"name": role, "description": "mirrors a mnemiq role; filled from Keycloak's groups"})
            groups[role] = made["id"]
    for profile, roles in PROFILES.items():
        model = {"id": profile, "name": profile, "base_model_id": None, "params": {}, "is_active": True,
                 "meta": {"description": "a mnemiq profile", "capabilities": CAPABILITIES},
                 "access_grants": [{"principal_type": "group", "principal_id": groups[r], "permission": "read"}
                                   for r in roles]}
        status, _ = request("POST", f"{base}/api/v1/models/create", token=admin, data=model)
        if status != 200:
            status, _ = request("POST", f"{base}/api/v1/models/model/update", token=admin, data=model)
        print(f"Open WebUI {option.upper()}: model {profile} for {', '.join(roles)} ({status})")
    ids: dict[str, str] = {}
    if option == "b":
        for user, role in OPTION_B_USERS.items():
            request("POST", f"{base}/api/v1/auths/add", token=admin, ok=(200, 400),
                    data={"name": user.title(), "email": f"{user}@example.test", "password": state[user],
                          "role": "user"})
            _, signed = request("POST", f"{base}/api/v1/auths/signin",
                                data={"email": f"{user}@example.test", "password": state[user]})
            ids[signed["id"]] = role
            request("POST", f"{base}/api/v1/groups/id/{groups[role]}/users/add", token=admin,
                    data={"user_ids": [signed["id"]]})
            print(f"Open WebUI B: {user} ({role}) ready")
    return ids


def cmd_users() -> int:
    if not STATE.exists():
        print("run `secrets` first")
        return 2
    state = json.loads(STATE.read_text())
    if not wait_for(f"{KEYCLOAK}/realms/{REALM}/.well-known/openid-configuration", "Keycloak"):
        return 1
    keycloak(state)
    if not wait_for(f"{OPEN_WEBUI['a']}/health", "Open WebUI A"):
        return 1
    open_webui(state, "a")
    try:
        urllib.request.urlopen(f"{OPEN_WEBUI['b']}/health", timeout=3)
    except (urllib.error.URLError, OSError):
        print("Open WebUI B is not running (it needs --profile option-b); skipped")
        return 0
    ids = open_webui(state, "b")
    write_private(SECRETS / "identity-map.json", json.dumps({"open-webui": {"sub": {i: [r] for i, r in ids.items()}}},
                                                            indent=1) + "\n", 0o644)
    print("identity map: option B's users mapped to their roles")
    return 0


def cmd_check() -> int:
    """Every service receives each secret it needs, read from Compose's resolved config inside this script: names
    are printed, values never are. Run under both profile sets, since `config` leaves out a service whose profile is
    not given."""
    failed = False
    for profiles in ([], ["--profile", "option-b"]):
        run = subprocess.run([*COMPOSE, *profiles, "config", "--format", "json"], capture_output=True, text=True)
        if run.returncode != 0:
            print(f"compose config failed ({' '.join(profiles) or 'no profile'}): {run.stderr.strip()[:300]}")
            failed = True
            continue
        services = json.loads(run.stdout)["services"]
        for service, needed in NEEDED.items():
            if service not in services:
                continue
            env = services[service].get("environment") or {}
            for key in needed:
                present = bool(env.get(key))
                failed |= not present
                print(f"{'ok' if present else 'MISSING'}: {service}: {key}")
    return 1 if failed else 0


if __name__ == "__main__":
    commands = {"secrets": cmd_secrets, "users": cmd_users, "check": cmd_check}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        print(__doc__)
        sys.exit(2)
    sys.exit(commands[sys.argv[1]]())
