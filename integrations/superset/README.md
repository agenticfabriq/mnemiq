# mnemiq extension for Superset

Superset extension `agenticfabriq.mnemiq` (Superset 7.0.0rc2). It lets mnemiq build and edit dashboards as the person
who asked. It carries a token route on Superset's web server and write tools on Superset's MCP service.

## Build

```
python scripts/build_superset_extension.py --out build/superset-extension
```

That writes `build/superset-extension/dist/` (for `LOCAL_EXTENSIONS`) and `mnemiq-0.1.0.supx` (for `EXTENSIONS_PATH`).
No dependency is installed: the extension uses `pyjwt` and `cryptography`, which Superset already ships.

## Install

In `superset_config.py`, then restart Superset (extensions load at startup):

```python
FEATURE_FLAGS = {"ENABLE_EXTENSIONS": True}
LOCAL_EXTENSIONS = ["/path/to/build/superset-extension"]   # a directory holding dist/
# or: EXTENSIONS_PATH = "/path/to/folder/of/supx/files"
```

Run `superset init` after the first load so roles receive the route's permission.

## Configuration

| Key | Meaning |
|---|---|
| `MNEMIQ_TOKEN_KEY_FILE` | The RSA private key (PEM) that signs tokens. A regular file with mode 0600. |
| `MNEMIQ_TOKEN_KID` | The `kid` in the token header, matching the key mnemiq and Superset's MCP service verify with. |
| `MNEMIQ_TOKEN_ISSUER` | The `iss` claim. |
| `MNEMIQ_TOKEN_AUDIENCE` | The `aud` claim. |
| `MNEMIQ_TOKEN_TENANT` | The `tenant` claim. |
| `MNEMIQ_TOKEN_LIFETIME_S` | Seconds a token lives. At most 300, which is also the default. |

All but the lifetime are required; the route answers 503 naming the one that is missing.

## The token route

`POST /extensions/agenticfabriq/mnemiq/token/` with `{"purpose": "ask"}` or `{"purpose": "build"}`, signed in, with
`X-CSRFToken`. It answers `{"token": "...", "expires_in": 300}`. The token is for the signed-in user only; the body
cannot name anyone else. An `ask` token carries `db.read superset:sqllab:write superset:dataset:read`; a `build` token
adds `superset:chart:read superset:chart:write superset:dashboard:read superset:dashboard:write superset:dataset:write`.

Superset's MCP service must verify these tokens: set its `MCP_JWT_PUBLIC_KEY` (or `MCP_JWKS_URI`), `MCP_JWT_AUDIENCE`
and `MCP_JWT_ISSUER` to match the key, audience and issuer above.

## The write tools

Registered on Superset's MCP service as `extensions.agenticfabriq.mnemiq.<name>`, each taking one `request` object
and running as the caller the token names. Each is gated by a stock permission with the method `write` (`can_write`
on Chart, Dataset or Dashboard) and the matching token scope (`superset:chart:write` and the like), so a role gets
from them what it already may do through Superset's own tools.

| Tool | Gated by | What it does |
|---|---|---|
| `create_dashboard` | Dashboard | An unpublished, empty dashboard with the caller as its editor. |
| `claim_chart` | Chart | Gives a chart a dashboard's editors and a build marker, and saves its query context. |
| `claim_dataset` | Dataset | Gives a dataset a dashboard's editors and a build marker. |
| `create_chart` | Chart | A funnel or a sankey, saved only once its query runs. |
| `patch_chart` | Chart | Sets and unsets named params keys, keeping every other key. |
| `add_saved_metric` | Dataset | Adds one aggregating saved metric, keeping the others. |
| `delete_dataset` | Dataset | Moves a dataset a build made to the trash once no chart uses it. |
| `set_layout` | Dashboard | Writes the layout, and with it the charts, keeping filters and tabs. |

Every write to an existing row first locks that row and compares the `changed_on` the caller read, so of two racing
writers the second answers `{"stale": true}`. Answers are short structured objects; Superset's own error text never
reaches them.
