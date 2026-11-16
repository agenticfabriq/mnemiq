# mnemiq extension for Superset

Superset extension `agenticfabriq.mnemiq` (Superset 7.0.0rc2). It lets mnemiq build and edit dashboards as the person
who asked. It carries a token route on Superset's web server, and write and read tools on Superset's MCP service.

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
| `MNEMIQ_LOCK_TIMEOUT_S` | Seconds a write tool waits for each lock another transaction holds before it answers `{"busy": true}`. Default 30. It must be above 0 and below 45, mnemiq's default deadline for a call, or a write could land after mnemiq gave up on it: any other value is not used, and the extension logs a warning naming the setting and waits 30. On a Postgres metadata database it bounds every lock wait; on MySQL it bounds a create's wait for its marker's lock (in whole seconds), and a row's lock waits as long as the database's own setting says. |

All but the lifetime and the lock timeout are required; the route answers 503 naming the one that is missing.

## The token route

`POST /extensions/agenticfabriq/mnemiq/token/` with `{"purpose": "ask"}` or `{"purpose": "build"}`, signed in, with
`X-CSRFToken`. It answers `{"token": "...", "expires_in": 300}`. The token is for the signed-in user only; the body
cannot name anyone else. An `ask` token carries `db.read superset:sqllab:write superset:dataset:read`; a `build` token
adds `superset:chart:read superset:chart:write superset:dashboard:read superset:dashboard:write superset:dataset:write`.
A token's `nbf` is five seconds before its `iat`, so an MCP service whose clock is a little behind the web server's
still takes it.

It mints only for a user signed in to Superset itself. An embedded guest (a request carrying a guest token, with
`EMBEDDED_SUPERSET` on) and a request with nobody signed in are answered 403, whatever role they hold: a guest's name is
whatever its guest token says, not a Superset sign-in, and the MCP service runs a token as the user of its name.

Superset's MCP service must verify these tokens: set its `MCP_JWT_PUBLIC_KEY` (or `MCP_JWKS_URI`), `MCP_JWT_AUDIENCE`
and `MCP_JWT_ISSUER` to match the key, audience and issuer above. If the MCP service also trusts an identity provider,
`MCP_JWT_ISSUER` lists both issuers, and Superset then refuses to start its MCP service until `MCP_USER_RESOLVER` names
a resolver: its default finds a user by the username or email a token names, whichever issuer signed it, so two
issuers' tokens for one username would be one user. Give it a resolver that binds the token's issuer.

## The write tools

Registered on Superset's MCP service as `extensions.agenticfabriq.mnemiq.<name>`, each taking one `request` object
and running as the caller the token names. Each is gated by a stock permission with the method `write` (`can_write`
on Chart, Dataset or Dashboard) and the matching token scope (`superset:chart:write` and the like), so a role gets
from them what it already may do through Superset's own tools.

| Tool | Gated by | What it does |
|---|---|---|
| `create_dashboard` | Dashboard | An unpublished, empty dashboard with the caller as its editor and a build marker. |
| `create_dataset` | Dataset | A dataset over a table, or over `sql` when given, made with its build marker and, as editors, the caller and the dashboard's users whose grants reach it. |
| `create_chart` | Chart | A chart of any family, from MCP's own chart config (and format keys to merge) or a funnel's or a sankey's form data, made with the dashboard's editors, a build marker and its query context, and saved only once its query runs. |
| `patch_chart` | Chart | Sets and unsets named params keys, keeping every other key. |
| `add_saved_metric` | Dataset | Adds one aggregating saved metric, keeping the others. |
| `delete_dataset` | Dataset | Moves a dataset a build made to the trash once no chart uses it. |
| `set_layout` | Dashboard | Writes the layout, and with it the charts, keeping filters and every tab under its id. |

### Who becomes an editor

In Superset 7.0.0rc2 an editor of a dataset may read its data: the access check admits a dataset's editors, and
SQL Lab admits an editor of any dataset over a table to that table. Editors are users, roles and groups, and a role or
a group made an editor makes every member one. So the tools choose editors with care: whether a user can already
read a dataset is checked when the tool adds them, and the extension never changes a dataset's source.

- A dataset that `create_dataset` makes for a dashboard, or finds by its build marker when sent again, keeps its
  editors (the tool removes none) and adds the caller and each user among the dashboard's editors whose grants already
  reach both the dataset and its table: datasource access to the dataset, or access to its database, to all databases
  or to all datasources. These are the grants SQL Lab's strict check of a table admits. Schema or catalog access alone
  does not count: it reads the dataset, but SQL Lab's strict check skips it, and there editorship of the dataset would
  reach the table. Each user is judged as that user, and being an editor of the dataset does not count. A role or a
  group among the dashboard's editors is never added.
- A chart that `create_chart` makes for a dashboard gets the dashboard's editors. When `ENABLE_VIEWERS` is on and
  `VIEWER_PROMISCUOUS_MODE` is set, a chart's editors may read its dataset's data, so a chart's editors are then chosen
  as a dataset's are, against the chart's dataset.

A co-editor of the dashboard who is not an editor of one of its datasets, and asks mnemiq to change or remove it, is
answered `{"not_editor": true, "editors": [...]}` naming the dataset's editors: whom to ask.

### Locks

Every write to an existing row first locks that row and compares the `changed_on` the caller read, so of two racing
writers the second answers `{"stale": true}`. A write that waits for the row longer than `MNEMIQ_LOCK_TIMEOUT_S`
answers `{"busy": true}` and changes nothing; the same request may be sent again. Answers are short structured objects;
Superset's own error text never reaches them.

A create carrying a build's marker (`create_dashboard`, `create_dataset`, `create_chart`) makes its object once, so a
create whose answer was lost may be sent again. Under a lock keyed by the marker (a transaction's advisory lock on
Postgres, a named lock on MySQL, each waited for at most `MNEMIQ_LOCK_TIMEOUT_S`, else `{"busy": true}`) it looks for
an object not in the trash whose description's last line is that marker: it answers that one's id; when there are
several it creates nothing and answers `{"candidates": [...]}`, the ids of those the caller may read (the list may be
empty); and it creates the object when there is none. `create_chart` also looks once before it runs the chart's
query, without the lock, so a chart already made is answered without running its query again; and when its config
or its query is refused, it looks again under the lock before it answers, so a chart that another send of the same
create made meanwhile is answered rather than refused, and one still being made answers `{"busy": true}`. On any
other metadata database than Postgres, MySQL or SQLite a create answers `{"refused": "metadata_database"}`. A SQLite
metadata database takes no marker lock, being Superset's single-person development default: there a create sent
again while the first is still running could make a second object.

On Postgres and MySQL the lookup needs the metadata database's READ COMMITTED isolation; under any other level a
create answers `{"refused": "isolation"}`. Superset 7.0.0rc2 sets READ COMMITTED itself only when
`SQLALCHEMY_ENGINE_OPTIONS` names no isolation level and `SQLALCHEMY_DATABASE_URI` is a `postgresql` or a `mysql` URI
(with any driver). A `mariadb://` URI, or options naming another level, need it set in `superset_config.py`:

```python
SQLALCHEMY_ENGINE_OPTIONS = {"isolation_level": "READ COMMITTED"}   # with any other options already set there
```

## The read tools

Registered beside the write tools, each gated by a stock permission with the method `read` (`can_read` on Chart,
Dataset or Dashboard) and the matching token scope, and changing nothing. An object the caller may not read is left out
of an answer, or answered as not found, exactly as one that does not exist. Every answer stays under 40,000 bytes of
JSON, inside Superset's MCP response size guard: an entry of a list that does not fit is answered with its id and
`"too_large": true`, and a layout that does not fit as `{"too_large": true}`.

| Tool | Gated by | What it answers |
|---|---|---|
| `dataset_metrics` | Dataset | Every saved metric of up to 20 datasets, with its currency and certification. |
| `editors_of` | Chart | The editors' labels of up to 100 charts, 20 datasets and 20 dashboards, for whom to ask for access, and for each dashboard whether the caller edits it. |
| `get_layout` | Dashboard | A dashboard's stored layout, as text, with the `changed_on` that `set_layout` takes back. |
