"""Superset's configuration for mnemiq's live contract suite: a throwaway Superset 7.0.0rc2 that mnemiq answers
through, as `SUPERSET_CONFIG_PATH` names it to every Superset process the suite runs against.

Nothing here is a value of its own. Every setting comes from the environment or from the keys directory the
environment names, and the keys are generated for each install by `setup_superset.py keys`, never committed:

- `MNEMIQ_TEST_SUPERSET_KEYS_DIR`: the test identity provider's public key (`public.pem`), the issuer and audience its
  tokens carry (`issuer.json`), and this Superset's own secret key (`secret_key`).
- `MNEMIQ_TEST_SUPERSET_METADATA_URI`: the SQLAlchemy URI of Superset's metadata database.
- `MNEMIQ_TEST_SUPERSET_DATA_CACHE_DIR`, optional: a directory for Superset's result cache. Unset, Superset keeps the
  default, which caches nothing.

Settings the suite relies on Superset's defaults for are left out on purpose, and named at the end.
"""

import json
import os
from pathlib import Path

_KEYS = Path(os.environ["MNEMIQ_TEST_SUPERSET_KEYS_DIR"])
_IDENTITY_PROVIDER = json.loads((_KEYS / "issuer.json").read_text())

# Signs Superset's sessions and keys the hashes it looks API keys up by. Generated per install, so no two installs
# share one, and Superset refuses to start on its shipped default.
SECRET_KEY = (_KEYS / "secret_key").read_text().strip()

# Superset's own users, roles, datasets and row rules. A server database rather than a file, so the web server, two
# MCP services and the setup script can each write to it at once.
SQLALCHEMY_DATABASE_URI = os.environ["MNEMIQ_TEST_SUPERSET_METADATA_URI"]

# The MCP service verifies each caller's token, so every tool runs as the user the token names, which is the contract
# under test: mnemiq forwards each person's own token and holds no credential of its own.
MCP_AUTH_ENABLED = True
# RS256, as an identity provider signs: Superset verifies with the public half only, and mnemiq's door verifies the same
# token against the same key, published as a key set.
MCP_JWT_ALGORITHM = "RS256"
MCP_JWT_PUBLIC_KEY = (_KEYS / "public.pem").read_text()
# One issuer, and the audience mnemiq's door checks too: the token a person signs in with is the one Superset accepts.
MCP_JWT_ISSUER = _IDENTITY_PROVIDER["issuer"]
MCP_JWT_AUDIENCE = _IDENTITY_PROVIDER["audience"]

# Superset's MCP service ships with tool search on, which lists a search tool and a call proxy in place of the tools
# themselves. Off here, so the catalogue a client lists names the tools it calls. mnemiq calls `execute_sql`,
# `list_datasets` and `get_database_info` by name, which Superset 7.0.0rc2 serves with tool search on as well.
from superset.mcp_service.mcp_config import MCP_TOOL_SEARCH_CONFIG as _TOOL_SEARCH  # noqa: E402

MCP_TOOL_SEARCH_CONFIG = {**_TOOL_SEARCH, "enabled": False}

# API keys on, so the service user enrichment reads through authenticates with an API key, as a deployment's would:
# the MCP service accepts them beside tokens when this is on (`get_mcp_api_key_enabled`).
FAB_API_KEY_ENABLED = True

# Superset's result cache, on a directory, for the MCP service the suite runs its cached tests against: a narrowed row
# rule and a new function must each be seen through a real cache, not only through none.
if _cache_dir := os.environ.get("MNEMIQ_TEST_SUPERSET_DATA_CACHE_DIR"):
    DATA_CACHE_CONFIG = {
        "CACHE_TYPE": "FileSystemCache",
        "CACHE_DIR": _cache_dir,
        "CACHE_DEFAULT_TIMEOUT": 24 * 60 * 60,
        "CACHE_KEY_PREFIX": "mnemiq_live_",
    }

# Left at Superset's defaults on purpose:
# - MCP_RESPONSE_SIZE_CONFIG: the response size guard a deployment runs with, which the suite's large result meets.
# - RLS_IN_SQLLAB, in FEATURE_FLAGS: off by default, which governs SQL Lab's REST path only; the MCP service applies
#   row rules to every statement whatever it says.
# - MCP_STRUCTURED_OUTPUT_ENABLED: off, so every answer arrives as JSON text, the shape a deployment sends by default.
# - MCP_CACHE_CONFIG: off, and never `dangerously_share_cache_across_principals`, which would answer one person's call
#   from another's.
