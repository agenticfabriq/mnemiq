# mnemiq extension for Superset

Superset extension `agenticfabriq.mnemiq` (Superset 7.0.0rc2). It lets mnemiq build and edit dashboards as the person
who asked. It carries a token route and a forwarder for mnemiq's panel on Superset's web server, and write and read tools
on Superset's MCP service.

## Build

```
python scripts/build_superset_extension.py --out build/superset-extension
```

That writes `build/superset-extension/dist/` (for `LOCAL_EXTENSIONS`) and `mnemiq-0.2.0.supx` (for `EXTENSIONS_PATH`).
No dependency is installed: the extension uses `pyjwt`, `cryptography` and `requests`, which Superset already ships.
When the panel's frontend is built (`integrations/superset/frontend/dist`), it is packed too; `--require-frontend`
refuses to pack without it. Every frontend file's name must carry a content hash, since Superset serves extension
files as immutable for a year, and no source map is packed.

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
| `MNEMIQ_URL` | Where the forwarder sends the panel's requests: mnemiq's address as Superset's web server reaches it, `http` or `https`, with no user, password, query or fragment (a path prefix is kept). The browser never needs to reach it. |
| `MNEMIQ_FORWARD_TIMEOUT_S` | Seconds a forwarded request may take in all, 1 to 900. Default 600: planning a build with a slow model can take minutes, so a proxy in front of Superset must allow as long. |
| `MNEMIQ_FORWARD_IDLE_S` | Seconds a forwarded chat answer's stream may go quiet, 5 to 300. Default 60. |
| `MNEMIQ_LOCK_TIMEOUT_S` | Seconds a write tool waits for each lock another transaction holds before it answers `{"busy": true}`. Default 30. It must be above 0 and below 45, mnemiq's default deadline for a call, or a write could land after mnemiq gave up on it: any other value is not used, and the extension logs a warning naming the setting and waits 30. On a Postgres metadata database it bounds every lock wait; on MySQL it bounds a create's wait for its marker's lock (in whole seconds), and a row's lock waits as long as the database's own setting says. So on MySQL set `innodb_lock_wait_timeout` (default 50 seconds) below 45: it bounds every row lock, a create's second lock among them. A longer wait can let a write land after mnemiq gave up on it: a create sent again then answers its object by its marker, while another write answers stale or busy. |

The token settings but the lifetime are required; the route answers 503 naming the one that is missing. The forwarder
needs `MNEMIQ_URL` and the token settings.

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

## The forwarder

`/extensions/agenticfabriq/mnemiq/v1/<path>`, signed in, with `X-CSRFToken` on a `POST`. It forwards the request to
mnemiq at `MNEMIQ_URL` with a token it mints for that request alone, for the signed-in user: an ask token for asking, a
build token for building. The browser never holds a mnemiq token, and Superset's content security policy needs no
change, since the panel calls only its own origin.

It forwards only these mnemiq routes, each by its method: `GET /v1/schema`, `POST /v1/chat` (streamed), `GET` and
`POST /v1/superset/builds` (the `GET` with `dashboard_id` alone), `GET /v1/superset/builds/<id>`, `POST
/v1/superset/builds/<id>/next`, `GET /v1/superset/dashboards/<id>/measures`, `POST
/v1/superset/dashboards/<id>/measures/<name>/edit|remove|save`, and `POST /v1/superset/dashboards/<id>/changes`.
mnemiq receives only `Authorization`, `Accept` and `Content-Type`, never Superset's cookie or CSRF header; the answer
carries back mnemiq's status, `Content-Type` and `WWW-Authenticate`, with `Cache-Control: no-store`. Neither a token
nor a body is logged.

What it refuses, before mnemiq is called, each with its sentence as `{"detail": ...}`:

| Status | When | Sentence |
|---|---|---|
| 403 | Nobody signed in, or an embedded guest | "the token route mints only for a signed-in Superset user" |
| 404 | A path that is not one of the routes above | "This extension does not forward that request to mnemiq." |
| 405 | One of those paths under another method | the same sentence |
| 400 | A query key the route does not take, or one given twice | "This extension does not forward that request's query to mnemiq." |
| 413 | A body over 64 KiB | "The request is larger than this extension forwards to mnemiq (65,536 bytes at most)." |
| 502 | mnemiq could not be reached | "mnemiq did not answer; try again, and if it keeps failing ask your Superset administrator to check mnemiq." |
| 504 | mnemiq did not answer in time | "mnemiq took too long to answer; try again, and if it keeps happening ask your Superset administrator to check mnemiq and the timeouts in front of it." |
| 503 | `MNEMIQ_URL` is not set or not valid | "This Superset's mnemiq extension does not know where mnemiq is: set MNEMIQ_URL in Superset's configuration." |

## The write tools

Registered on Superset's MCP service as `extensions.agenticfabriq.mnemiq.<name>`, each taking one `request` object
and running as the caller the token names. Each is gated by a stock permission with the method `write` (`can_write`
on Chart, Dataset or Dashboard) and the matching token scope (`superset:chart:write` and the like), so a role gets
from them what it already may do through Superset's own tools.

| Tool | Gated by | What it does |
|---|---|---|
| `create_dashboard` | Dashboard | An unpublished, empty dashboard with the caller as its editor and a build marker. |
| `create_dataset` | Dataset | A dataset over a table, or over `sql` when given, made with its build marker and, as editors, the caller and the dashboard's users whose grants reach it. |
| `create_chart` | Chart | A chart of any family, from MCP's own chart config (and format keys to merge, and up to 5 SQL HAVING conditions to add to its filters) or a funnel's or a sankey's form data, made with the dashboard's editors, a build marker and its query context, and saved only once its query runs. |
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
writers the second answers `{"stale": true}`. A write that waits for the row longer than `MNEMIQ_LOCK_TIMEOUT_S` answers
`{"busy": true}` and changes nothing; the same request may be sent again. So does a write whose connection to the
metadata database was lost mid-statement, that found no free connection in the pool, or whose statement the metadata
database canceled (a `statement_timeout` shorter than the lock timeout cuts a wait short with SQLSTATE 57014): it either
did not write or cannot tell whether it did, and a create sent again finds what an earlier send made by its build
marker. Any other database error is answered `{"failed": true}`; a write tool never raises one. Answers are short
structured objects; Superset's own error text never reaches them.

A create carrying a build's marker (`create_dashboard`, `create_dataset`, `create_chart`) makes its object once, so a
create whose answer was lost may be sent again. Once the caller passes the create's own checks (the request, the dataset
and dashboard it names, being an editor of the dashboard), the create takes one lock keyed by the marker, on a metadata
database connection of its own (a session's advisory lock on Postgres, a named lock on MySQL), and holds it until the
object is made and committed. Under it, the create looks for an object not in the trash whose description's last line is
that marker: it answers that one's id; when there are several it creates nothing and answers `{"candidates": [...]}`,
the ids of those the caller may read (the list may be empty); and when there is none, it runs Superset's own checks (a
chart's config validation and its query, a dataset's table; a histogram's query is the extension's own, because Superset
7.0.0rc2's compile check leaves a histogram's column out of its query) and creates the object, a dataset's editors included. So a
chart already made is answered without running its query again, and a create sent again while the first is still running
waits for the first and answers what it made. A second lock guards the create itself: just before its command, in
the command's own transaction, the create takes a second lock (on Postgres a transaction's lock on another key of the
marker, on MySQL a row lock), looks again, and confirms on its marker lock's own connection that the lock is still held
(the hold check, a short transaction of its own on that connection, which does not end the lock). So if the first
lock's connection was lost while the create worked (its lock goes with it), an object another send made meanwhile is
found and answered rather than made twice, and a create whose lock was lost answers `{"busy": true}` and makes
nothing: it never takes the lost lock again, and the extension logs a warning that the marker's lock was lost while
the create worked. A hold check that fails with its connection still open (a statement the metadata database
canceled, say) cannot tell whether the lock is held: the create answers `{"busy": true}` and makes nothing too, and the
warning says instead that the create could not confirm its marker's lock is held; the lock is then let go as usual.
When Superset's checks or the create's command answer anything but the object or `{"busy": true}` (a refusal or a
failure), or raise anything (a database error, or a refusal such as the one for a table whose dataset is in the
trash, which is answered by the caller's access once the marker's lock is let go), the create first takes the second
lock and looks once more, with the hold check (the settle), so that answer never stands while another send of the
marker is making the object: it answers that object instead, or `{"busy": true}` when its own lock was lost. Only the
answers of setting the editors of a dataset already made or found are given without a settle. On MySQL the
second lock is a row lock: the dashboard's row for a chart or a dataset, and the caller's own user row for a
dashboard. It is taken just before the command, and again by the settle, and held until that transaction ends (the
command's commit, or the rollback that ends the request), so every other write that locks the same row waits for it:
an edit of that dashboard (`set_layout` among them), another create on it, or another dashboard create by that person.
For a dataset that time includes the command's read of the source table's columns, so a slow source database holds up
those writes, each for at most the metadata database's `innodb_lock_wait_timeout`, after which it answers
`{"busy": true}`. While a dashboard create holds its caller's user row, any insert that references that user by a
foreign key waits for it too. A dashboard deleted while a create for it works leaves no row to lock: the create
answers `{"busy": true}`, and the same create sent again is answered `{"not_found": "dashboard"}`. A wait for the
marker's lock longer than `MNEMIQ_LOCK_TIMEOUT_S` or cut short by the metadata database's `statement_timeout`, no
free connection in the metadata database's pool to take it on, or a look the metadata database cannot answer (the
first, the one just before the command, or the settle's), answers `{"busy": true}`: the same request may be sent
again. A create whose lock comes free more than 45 seconds after the create began (mnemiq's default deadline for a
call) lets it go at once and answers
`{"busy": true}`, since mnemiq has by then stopped waiting for the answer and sends the create again. Where mnemiq is
set to wait longer than 45 seconds for a call, such a create still answers busy, and is then sent again. While it waits
for the lock and while it holds it, a create uses two of the metadata database's pooled connections. On any other
metadata database than Postgres, MySQL or SQLite a create answers `{"refused": "metadata_database"}`. A SQLite metadata
database takes no marker lock, being Superset's single-person development default: there a create sent again while the
first is still running could make a second object.

On Postgres the marker's lock belongs to one database session, the one its connection opened. So Superset must reach a
Postgres metadata database directly, or through a pooler that keeps one server session for each client connection
(PgBouncer's `pool_mode = session`). Behind transaction pooling (`pool_mode = transaction`) the server session that took
the lock goes back to the pooler when the lock's transaction commits, and later statements of the create's connection,
its release among them, may reach other sessions: a second send of a create then runs its own checks rather than waiting
for the first (only the create's own transaction lock keeps it from making a second object), and a marker's lock can
stay held after its create has ended. The hold check's short transaction may reach another server session too, so a
create may answer `{"busy": true}` though its lock is held, or count as held a lock another session holds. Transaction
pooling is not supported.

On Postgres and MySQL alike, the marker lock's connection sits idle from the moment the lock is taken until each hold
check: through the look, Superset's own checks, and, for a settle after the command refused, the command itself (a
dataset's command reads its table's columns from the source database). A timeout that closes an idle connection
(Postgres's `idle_session_timeout`, MySQL's `wait_timeout`, a proxy's or a load balancer's) shorter than that closes
the lock's connection while the create works, so such a create answers `{"busy": true}` every time it is sent, and
the extension logs each time that the marker's lock was lost while its create worked. The timeout must be longer
than the longest a create works under its lock: a chart's query check, or a dataset's read of its table's columns.

On Postgres and MySQL the lookup needs the metadata database's READ COMMITTED isolation; under any other level every
create answers `{"refused": "isolation"}`, a create of a chart already made included. Superset 7.0.0rc2 sets READ
COMMITTED itself only when `SQLALCHEMY_ENGINE_OPTIONS` names no isolation level and `SQLALCHEMY_DATABASE_URI` is a
`postgresql` or a `mysql` URI (with any driver). A `mariadb://` URI, or options naming another level, need it set in
`superset_config.py`:

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
| `editors_of` | Chart | The editors' labels of up to 100 charts, 20 datasets and 20 dashboards, for whom to ask for access, and for each one whether the caller edits it. |
| `get_layout` | Dashboard | A dashboard's stored layout, as text, with the `changed_on` that `set_layout` takes back. |
| `row_rules` | Dataset | For each of up to 20 datasets, whether a row rule attached to it applies to the caller, as Superset's `get_rls_filters` finds it (an embedded dashboard's guest rules are not counted); never which rule, its clause or its roles. |
