# mnemiq extension for Superset

Superset extension `agenticfabriq.mnemiq` (Superset 7.0.0rc2). It lets mnemiq build and edit dashboards as the person
who asked. This first version carries the token route; the write tools follow.

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
