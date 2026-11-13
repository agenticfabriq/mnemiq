"""Fixtures for the live contract suite: mnemiq in Superset mode against a real Superset 7.0.0rc2 and the Postgres
database behind it, set up as `setup_superset.py` describes.

The suite is skipped unless these four are set, and the skip names them:

- `MNEMIQ_TEST_SUPERSET_URL`: Superset's web server, asked only whether it is up.
- `MNEMIQ_TEST_SUPERSET_MCP_URL`: Superset's MCP service, with Superset's default result cache, which caches nothing.
- `MNEMIQ_TEST_SUPERSET_PG_DSN`: the source database behind Superset, seeded from `seed.sql`. mnemiq in Superset mode
  never reads it; the suite reads it for the ground truth each answer is held to, and the direct enrichment L1
  compares against reads it as a direct deployment would.
- `MNEMIQ_TEST_SUPERSET_KEYS_DIR`: what `setup_superset.py keys` and `setup_superset.py` wrote: the test identity
  provider's private key, key set, issuer and audience, and the service user's API key.

Some tests need more, and fail naming what is missing rather than skip, so a run meant to be live never passes by
skipping part of itself:

- `MNEMIQ_TEST_SUPERSET_CACHED_MCP_URL`: a second MCP service on the same Superset, with its result cache on a directory
  (`MNEMIQ_TEST_SUPERSET_DATA_CACHE_DIR` in `superset_config.py`).
- `MNEMIQ_TEST_SUPERSET_PYTHON` and `MNEMIQ_TEST_SUPERSET_METADATA_URI`: Superset's interpreter and its metadata
  database, for `setup_superset.py narrow` and `widen`, which change a row rule while the suite runs.

No model or embedder is reached: the model is the stub in `tests/sweep/stub_model.py` on 127.0.0.1, which answers the
writer with the SQL a test sets, enrichment's column notes with its own text, and embeddings with vectors derived from
the text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.superset_live.live import DATABASE_NAME, GATE, Live, enrich_and_build, record_calls, superset_settings


@pytest.fixture(scope="session")
def live() -> Live:
    missing = [name for name in GATE if not os.environ.get(name, "").strip()]
    if missing:
        pytest.skip(f"the live Superset suite needs {', '.join(missing)} (see tests/superset_live/conftest.py)")
    import httpx

    url, mcp_url, pg_dsn, keys = (os.environ[name].strip() for name in GATE)
    health = httpx.get(f"{url.rstrip('/')}/health", timeout=10)
    assert health.status_code == 200, f"Superset's web server is not up: HTTP {health.status_code}"
    keys = Path(keys)
    provider = json.loads((keys / "issuer.json").read_text())
    return Live(url=url, mcp_url=mcp_url, pg_dsn=pg_dsn, keys=keys, issuer=provider["issuer"],
                audience=provider["audience"], kid=provider["kid"], database_id=_database_id(mcp_url, keys))


def _database_id(mcp_url: str, keys: Path) -> int:
    """The id Superset gave the source database, asked of Superset as the service user."""
    from mnemiq.superset.client import SupersetMcp
    from mnemiq.superset.service import ServiceKey

    client = SupersetMcp(mcp_url, ServiceKey((keys / "svc.key").read_text().strip()))
    listed = client.call("list_databases", {"filters": [{"col": "database_name", "opr": "eq",
                                                         "value": DATABASE_NAME}]})
    ids = [item["id"] for item in listed.get("databases", []) if item.get("database_name") == DATABASE_NAME]
    assert len(ids) == 1, f"Superset lists {len(ids)} databases named {DATABASE_NAME}: run setup_superset.py"
    return ids[0]


@pytest.fixture(scope="module", autouse=True)
def _no_mnemiq_environment():
    """`Settings()` reads every `MNEMIQ_` variable, matching names in any case, so one set in the shell running the
    suite must not decide them. The suite's own `MNEMIQ_TEST_SUPERSET_` variables, and the store path the root conftest
    sets, stay. Module-scoped, as the fixtures that build settings are, so it is in place before any of them and gone
    once the suite's module ends."""
    keep = ("MNEMIQ_TEST_SUPERSET_", "MNEMIQ_STORE_PATH")
    with pytest.MonkeyPatch.context() as patch:
        for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and not n.upper().startswith(keep)]:
            patch.delenv(name)
        yield


@pytest.fixture(scope="module")
def model():
    """The stub model. The writer answers with whatever SQL a test sets on `model.sql`."""
    from tests.sweep.stub_model import StubModel

    with StubModel(sql="SELECT 1") as stub:
        yield stub
    assert not stub.all_errors()


@pytest.fixture(scope="module")
def enriched(live, model, tmp_path_factory):
    """The settings of the store every Superset-mode test answers from, enriched through Superset as the service user
    and built, and what enriching and building it printed."""
    settings = superset_settings(live, model, tmp_path_factory.mktemp("superset-store") / "store.duckdb")
    return settings, enrich_and_build(settings, live.service_key_file)


@pytest.fixture
def recorder(monkeypatch):
    """Each call mnemiq makes to Superset while the test runs, through the real client."""
    return record_calls(monkeypatch)
