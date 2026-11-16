"""What the live contract suite shares beside its fixtures: the Superset it runs against, the settings of the two
deployments it compares, enrichment as the CLI runs it, the ground truth read from the source, the setup script, and a
recorder of every call mnemiq makes to Superset. `conftest.py` says which environment variables describe the Superset.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest

HERE = Path(__file__).parent
# The variables the suite is skipped without.
GATE = ("MNEMIQ_TEST_SUPERSET_URL", "MNEMIQ_TEST_SUPERSET_MCP_URL", "MNEMIQ_TEST_SUPERSET_PG_DSN",
        "MNEMIQ_TEST_SUPERSET_KEYS_DIR")
SCHEMA = "public"
SOURCE_ID = "live"
TENANT = "live"
# The database `setup_superset.py` registers in Superset.
DATABASE_NAME = "mnemiq_live"
# What each person's token carries: mnemiq's door requires `db.read`, and Superset 7.0.0rc2 holds a token that carries
# any scope to its own scope map (`_token_scope_allows` in `superset/mcp_service/auth.py`), so the token also names
# Superset's scopes for `execute_sql` and `list_datasets`.
SCOPES = "openid db.read superset:sqllab:write superset:dataset:read"
# The extension's token route (`integrations/superset`), under the prefix Superset gives an extension's routes.
TOKEN_ROUTE = "/extensions/agenticfabriq/mnemiq/token/"


def required(*names: str) -> dict[str, str]:
    """The values of `names` from the environment; the test fails naming any that is not set, rather than skip."""
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        pytest.fail(f"the live suite needs {' and '.join(missing)} for this test (see tests/superset_live/conftest.py)")
    return {name: os.environ[name].strip() for name in names}


@dataclass(frozen=True)
class Live:
    """The Superset the suite runs against, as the environment describes it."""

    url: str
    mcp_url: str
    pg_dsn: str
    keys: Path
    issuer: str
    audience: str
    kid: str
    database_id: int

    def token(self, username: str, *, scope: str = SCOPES, **claims) -> str:
        """A token the test identity provider signs for `username`, which mnemiq's door and Superset both accept."""
        import jwt

        now = int(time.time())
        body = {"iss": self.issuer, "aud": self.audience, "sub": username, "preferred_username": username,
                "iat": now, "nbf": now, "exp": now + 900, "jti": uuid.uuid4().hex, "tenant": TENANT, "scope": scope,
                **claims}
        return jwt.encode(body, (self.keys / "private.pem").read_bytes(), algorithm="RS256",
                          headers={"kid": self.kid})

    @contextlib.contextmanager
    def signed_in(self, username: str):
        """An HTTP client signed in to Superset's web server as `username` through `/api/v1/security/login`, with the
        password `setup_superset.py` keeps for that user, and the Bearer the sign-in answered; closed when the block
        ends. The password is read here and sent nowhere else; no message this raises carries it."""
        import httpx

        password = json.loads((self.keys / "passwords.json").read_text())[username]
        with httpx.Client(base_url=self.url, timeout=30) as client:
            login = client.post("/api/v1/security/login",
                                json={"username": username, "password": password, "provider": "db", "refresh": False})
            assert login.status_code == 200, f"Superset refused {username}'s sign-in: HTTP {login.status_code}"
            yield client, {"Authorization": f"Bearer {login.json()['access_token']}"}

    def csrf_token(self, client, headers: dict) -> str:
        """A CSRF token for this signed-in client, from `/api/v1/security/csrf_token/`; its session cookie stays on
        the client."""
        answer = client.get("/api/v1/security/csrf_token/", headers=headers)
        assert answer.status_code == 200, f"no CSRF token: HTTP {answer.status_code}"
        return answer.json()["result"]

    def minted(self, username: str, purpose: str) -> str:
        """A token the extension's token route mints for `username`, signed in, with a CSRF token: what mnemiq's
        panel asks for on a person's behalf."""
        with self.signed_in(username) as (client, headers):
            csrf = self.csrf_token(client, headers)
            answer = client.post(TOKEN_ROUTE, json={"purpose": purpose}, headers={**headers, "X-CSRFToken": csrf})
        assert answer.status_code == 200, f"the token route answered HTTP {answer.status_code}"
        return answer.json()["token"]

    @property
    def jwks_file(self) -> str:
        return str(self.keys / "jwks.json")

    @property
    def service_key_file(self) -> str:
        return str(self.keys / "svc.key")


def superset_settings(live: Live, model, store: Path, *, mcp_url: str | None = None, **over):
    """A Superset-mode deployment over `store`, answering through `mcp_url` (the uncached MCP service by default),
    whose door verifies the test identity provider's tokens."""
    from mnemiq.config import Settings

    return Settings(acme_data_dir=None, llm_base_url=model.url, llm_api_key="stub-key", llm_model="stub-model",
                    embed_base_url=model.url, embed_api_key="stub-key", embed_model="stub-embed",
                    store_path=str(store), source_id=SOURCE_ID, answer_log_path="off", llm_window_check=False,
                    superset_mcp_url=mcp_url or live.mcp_url, superset_database_id=live.database_id,
                    superset_schema=SCHEMA, oidc_issuer=live.issuer, oidc_audience=live.audience,
                    oidc_jwks_file=live.jwks_file, tenant=TENANT, **over)


def direct_settings(live: Live, model, store: Path, **over):
    """A direct deployment over `store`, reading the same Postgres with a connection of its own."""
    from mnemiq.config import Settings

    return Settings(acme_data_dir=None, llm_base_url=model.url, llm_api_key="stub-key", llm_model="stub-model",
                    embed_base_url=model.url, embed_api_key="stub-key", embed_model="stub-embed",
                    store_path=str(store), source_id=SOURCE_ID, answer_log_path="off", llm_window_check=False,
                    **{"pg_dsn": live.pg_dsn, **over})


def enrich_and_build(settings, service_key_file: str | None) -> str:
    """Run `mnemiq enrich` then `mnemiq build` on `settings`, as the CLI runs them, with the service key file named
    as an operator names it when one is given; return what they printed. Each must exit 0."""
    from mnemiq.cli import _cmd_build, _cmd_enrich
    from mnemiq.superset.service import SERVICE_KEY_ENV

    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        if service_key_file is None:
            patch.delenv(SERVICE_KEY_ENV, raising=False)
        else:
            patch.setenv(SERVICE_KEY_ENV, service_key_file)
        enriched = _cmd_enrich(settings)
        built = _cmd_build(settings) if enriched == 0 else None
    printed = out.getvalue() + err.getvalue()
    assert (enriched, built) == (0, 0), printed
    return printed


def ground_truth(live: Live, sql: str, params: tuple = ()) -> list[tuple]:
    """`sql`'s rows read from the source database directly: by the suite, to hold an answer to, and never by mnemiq."""
    import psycopg

    with psycopg.connect(live.pg_dsn, connect_timeout=10) as con:
        return [tuple(row) for row in con.execute(sql, params).fetchall()]


def run_setup(*args: str) -> str:
    """`setup_superset.py` with `args`, run by Superset's own interpreter; the last line it printed."""
    env = required("MNEMIQ_TEST_SUPERSET_PYTHON", "MNEMIQ_TEST_SUPERSET_METADATA_URI")
    done = subprocess.run([env["MNEMIQ_TEST_SUPERSET_PYTHON"], str(HERE / "setup_superset.py"), *args],
                          env={**os.environ, "SUPERSET_CONFIG_PATH": str(HERE / "superset_config.py")},
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip().splitlines()[-1]


def metadata(sql: str, params: tuple = ()) -> list[tuple]:
    """`sql`'s rows read from Superset's metadata database directly: by the suite, to see what a tool wrote, and never
    by mnemiq or the extension."""
    import psycopg

    uri = required("MNEMIQ_TEST_SUPERSET_METADATA_URI")["MNEMIQ_TEST_SUPERSET_METADATA_URI"]
    with psycopg.connect(uri, connect_timeout=10) as con:
        return [tuple(row) for row in con.execute(sql, params).fetchall()]


def carrying(table: str, marker: str) -> list[tuple[int, object]]:
    """Every row of the metadata database's `table` (`slices`, `tables` or `dashboards`), in the trash or not, whose
    description's last line is exactly `marker`, as `(id, deleted_at)` by id: what a create's lookup for the marker
    finds, the trash included. `strpos` reads the marker as text, where a `LIKE` would read a step key's `_` as any
    one character."""
    rows = metadata(f"SELECT id, description, deleted_at FROM {table} WHERE strpos(description, %s) > 0 ORDER BY id",
                    (marker,))
    return [(row[0], row[2]) for row in rows if (row[1] or "").splitlines()[-1:] == [marker]]


def fingerprint_of(value: str) -> str:
    """A short digest of a credential, so a test can tell two apart without keeping either."""
    return hashlib.sha256(value.encode()).hexdigest()[:12]


@dataclass
class Call:
    """One `SupersetMcp.call`: the tool, what was sent, a fingerprint of the credential, how long it took, the JSON-RPC
    requests its session made with how long each took, and what it returned or raised."""

    tool: str
    request: dict
    credential: str
    seconds: float = 0.0
    rpc: list[tuple[str, float]] = field(default_factory=list)
    payload: dict | None = None
    raised: BaseException | None = None

    @property
    def sql(self) -> str:
        return self.request.get("sql", "")

    @property
    def answers(self) -> bool:
        """Whether this call ran a statement for its rows: not a dry run, a catalogue read or the proof's probe."""
        return (self.tool == "execute_sql" and not self.request.get("dry_run")
                and not self.request.get("force_refresh") and not self.sql.endswith("AS mnemiq_probe WHERE 1 = 0"))


class Recorder:
    """Every `SupersetMcp.call` made while a test runs, in order, each with the JSON-RPC requests its session made."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.current = threading.local()
        self.lock = threading.Lock()

    def mark(self) -> int:
        return len(self.calls)

    def since(self, mark: int) -> list[Call]:
        return self.calls[mark:]


def record_calls(monkeypatch) -> Recorder:
    """Wrap, never replace, the real client so each call it makes is recorded: `SupersetMcp.call` for the call, and the
    MCP session's `send_request` for each JSON-RPC request inside it."""
    import mcp

    from mnemiq.superset.client import SupersetMcp

    recorded = Recorder()
    real_call = SupersetMcp.call
    real_send = mcp.ClientSession.send_request

    def call(self, tool, request, **kwargs):
        record = Call(tool, dict(request), f"{type(self.credential).__name__}:{fingerprint_of(self.credential.value)}")
        with recorded.lock:
            recorded.calls.append(record)
        recorded.current.call = record
        started = time.perf_counter()
        try:
            record.payload = real_call(self, tool, request, **kwargs)
            return record.payload
        except BaseException as exc:
            record.raised = exc
            raise
        finally:
            record.seconds = time.perf_counter() - started
            recorded.current.call = None

    async def send_request(self, request, *args, **kwargs):
        started = time.perf_counter()
        try:
            return await real_send(self, request, *args, **kwargs)
        finally:
            record = getattr(recorded.current, "call", None)
            if record is not None:
                record.rpc.append((request.root.method, time.perf_counter() - started))

    monkeypatch.setattr(SupersetMcp, "call", call)
    monkeypatch.setattr(mcp.ClientSession, "send_request", send_request)
    return recorded
