# mnemiq with Superset

In Superset mode mnemiq answers each builder's questions through [Apache Superset](https://superset.apache.org)'s MCP
service, as that builder. A builder is anyone who asks mnemiq questions. Every statement runs with the token the
builder's request carries, so Superset's roles and row rules decide what it reads, and mnemiq holds no database
credential of its own. Enrichment reads the source offline through Superset too, as a service user whose API key only
the command line reads. This page covers what each side needs, what mnemiq refuses, and what each refusal tells the
person asking.

It was checked with **Superset 7.0.0rc2** on PostgreSQL 16, by the contract suite in
[`tests/superset_live`](../tests/superset_live). That suite's
[`superset_config.py`](../tests/superset_live/superset_config.py) is a working Superset configuration, except that it
leaves Superset's response size guard at its default on purpose, so that the suite meets a result the guard cuts;
change that setting as [What Superset needs](#what-superset-needs) says. Where this page says "measured", the fact was
measured against that Superset.

## How it answers

- **Who asks.** Builders, on `/v1`: the workbench and the HTTP API. mnemiq's door verifies each request's Bearer token
  as it always does, then hands the same token, unchanged, to Superset on every call that request makes. `/v1/whoami`
  asks Superset nothing.
- **What they may read.** For each request mnemiq asks Superset, as the builder, which of the snapshot's tables they
  may read, with a dry run of a statement that names them, and offers the model only those. mnemiq keeps no grants of
  its own. Superset has no column rules for SQL, so mnemiq masks no column: every column of a table a builder may read
  is theirs to read.
- **What it sends.** Each statement goes to `execute_sql` with the database, the schema, the statement and a time
  limit. mnemiq never sends `limit` or `template_params`. A statement holding `{{`, `{%` or `{#` is never sent,
  since Superset would render it as a template. Superset applies its row rules to each statement before it runs it.
- **Caching.** mnemiq keeps no result cache in this mode. Superset's own result cache (`DATA_CACHE_CONFIG`) may stay
  on: it keys a result on the statement after row rules are applied, so a narrowed row rule never meets an old entry
  (measured, with a file cache), and `execute_sql` checks access before it looks, so a revoked grant is refused before
  any entry is read. Which functions the database defines is read afresh for every request, and those catalogue
  statements ask Superset to run them rather than answer from its cache.
- **Follow-ups.** Rows a client sends back with an earlier turn never reach the prompt: they may be someone else's, or
  older than what the builder may read now. The newest earlier turn with SQL is decided again under the builder's
  current grants and run again as them.
- **Read-only, and no `deep`.** Superset mode makes no write, and offers the `instant` and `thinking` modes only.

## What Superset needs

- **Superset 7.0.0rc2.**
- **A PostgreSQL database, version 12 or later.** Superset mode reads PostgreSQL only, and reads
  `pg_attribute.attgenerated`, which PostgreSQL 12 added.
- **MCP auth on, verifying your identity provider's tokens with RS256.** One token serves both sides: mnemiq's door
  checks it, then passes it on, so Superset must accept the issuer and audience mnemiq does.

  | Superset setting | Value |
  |---|---|
  | `MCP_AUTH_ENABLED` | `True` |
  | `MCP_JWT_ALGORITHM` | `"RS256"` |
  | `MCP_JWT_PUBLIC_KEY` | your provider's signing key, as PEM (Superset also accepts a key set at `MCP_JWKS_URI`) |
  | `MCP_JWT_ISSUER` | the same as `MNEMIQ_OIDC_ISSUER` |
  | `MCP_JWT_AUDIENCE` | the same as `MNEMIQ_OIDC_AUDIENCE` |

- **API keys on** (`FAB_API_KEY_ENABLED = True`), for the service user enrichment reads as. The MCP service accepts
  API keys while this is on, unless `MCP_API_KEY_ENABLED` says otherwise.
- **The tools mnemiq calls left on.** `execute_sql`, `list_datasets` and `get_database_info` must not be in
  `MCP_DISABLED_TOOLS`. Tool search (`MCP_TOOL_SEARCH_CONFIG`) may stay on: Superset 7.0.0rc2 serves a tool called by
  name either way (measured).
- **Results sent whole.** Superset's response size guard (`MCP_RESPONSE_SIZE_CONFIG`, 50,000 bytes by default) cuts
  a larger `execute_sql` result by dropping rows. mnemiq never answers from a cut result: it asks the model to
  aggregate or select less, and if the model cannot, the builder reads "Could not answer this question: the database
  rejected every attempt. The source rejected this query.", which does not say the result was too large. So a large
  result is never answered unless you take `execute_sql` out of the guard, which is the better fix, or raise its
  `max_bytes` to about twice the size of the largest result you want answered: the guard measures the whole answer,
  and `execute_sql`'s answer carries every row twice, once as the result's rows and again in the statement's own data.
  To take `execute_sql` out, add it to the guard's `excluded_tools`:

  ```python
  from superset.mcp_service.mcp_config import MCP_RESPONSE_SIZE_CONFIG as _SIZE_GUARD

  MCP_RESPONSE_SIZE_CONFIG = {**_SIZE_GUARD, "excluded_tools": [*_SIZE_GUARD["excluded_tools"], "execute_sql"]}
  ```

- **Structured output off** (`MCP_STRUCTURED_OUTPUT_ENABLED = False`, the default). mnemiq was checked against the
  JSON text Superset sends with it off.
- **The MCP response cache off** (`MCP_CACHE_CONFIG`, off by default), and **never
  `dangerously_share_cache_across_principals`**. That cache serves a hit before Superset checks who is asking, so one
  builder's call could be answered with another builder's result.

## What PostgreSQL needs

**A `statement_timeout`** on the role, or the connection, Superset uses for this database. Superset applies a
statement's own time limit only once the database call returns: it reports a slow statement as timed out when the
statement ends, and never stops it. Measured: a statement given 1 s ran to its end, about 6.4 s, before Superset
reported it timed out, and a longer one went on running in PostgreSQL after mnemiq stopped waiting for it. Only
PostgreSQL's `statement_timeout` stops a statement.

mnemiq gives each statement it runs for a builder a time limit of 30 s at most, and stops waiting for Superset's answer
15 s after that limit, so at 45 s at the latest. It reads a statement still running then as that statement's cost, and
asks the model for a cheaper one, as it does when Superset reports a statement timed out. Enrichment gives each of its
statements 300 s, and waits 315 s.

Enrichment reads through the same Superset database builders ask through, so unless Superset impersonates each user
on that database (off by default), both reach PostgreSQL as the one role its connection names, and one
`statement_timeout` holds both. Choose it knowing the trade-off:

- **Near 45 s,** a builder's statement stops soon after mnemiq stops waiting for it, but enrichment can be stopped on a
  large table. A table it cannot count the rows of, or any column of, is left out of the snapshot, and builders cannot
  ask about it; a column it cannot count is kept without counts. `mnemiq enrich` names each.
- **Near 300 s,** enrichment's statements get the whole time mnemiq gives them, but a builder's statement can go on
  running for up to that long after mnemiq stopped waiting for it, holding the database and a process of Superset's
  MCP service ([Capacity](#capacity)).

## What your identity provider puts in a builder's token

- **An `iss` and an `aud`** that mnemiq and Superset both accept (above), and a **`kid`** in the header naming the
  signing key, which mnemiq's door looks up in the provider's key set.
- **`preferred_username`**, the builder's Superset username. Superset's default user resolver reads it first, then
  `username`, `email` and `sub`, and refuses a token whose user Superset has no account for.
- **The tenant claim** (`MNEMIQ_OIDC_TENANT_CLAIM`, `tenant` by default) holding `MNEMIQ_TENANT`.
- **Scopes:** `db.read`, which mnemiq's door requires, and Superset's own, `superset:sqllab:write` and
  `superset:dataset:read`, or the flat `superset:write` and `superset:read`. When a token carries any scope, Superset
  7.0.0rc2 allows a tool only if both the token's scopes and the builder's roles permit it, and `db.read` means every
  token mnemiq passes on carries one. `execute_sql` needs `superset:sqllab:write`, and `list_datasets` needs
  `superset:dataset:read`. A token without them is refused with the text Superset sends for a missing role, so the
  builder reads the roles sentence below although they may hold both roles (measured).

The workbench reaches `/v1` the usual way, behind oauth2-proxy injecting the access token, as
[the Open WebUI guide](open-webui.md#the-workbench-behind-oauth2-proxy) sets it up. Its client at your provider
requests these scopes too. Open WebUI's own endpoint is not served in Superset mode.

## Roles in Superset

- **Builders need Superset's Alpha and sql_lab roles.** Alpha gives access to every database, and mnemiq needs access
  to the whole database it answers from. A builder without sql_lab, or without the scopes above, reads: "Superset
  refused this user's queries. Building with mnemiq needs Superset's Alpha and sql_lab roles."
- **A builder granted only datasets, not the database, cannot ask yet.** Superset does not give them PostgreSQL's
  catalogue, so mnemiq cannot read which functions the database defines, and cannot prove any statement free of a
  function call. It declines every question, running nothing, with: "This source could not say which functions it
  defines, so this engine cannot confirm what this query executes against it." Asking with access to datasets alone is
  not supported yet.
- **The service user** enrichment reads as has **Alpha** and **sql_lab**, and **no row rules**. What enrichment
  measures (counts, and which keys are unique) must hold for every builder, and each builder sees the service user's
  rows or fewer.

## Row rules

- **Attach each row rule to the dataset of every physical table people can read.** Superset applies row rules to a
  statement through the physical tables it reads, each table taking the rules attached to its own dataset, and mnemiq's
  statements, like a SQL Lab query, read physical tables. A rule attached only to a virtual dataset applies through that
  dataset, as in SQL Lab: it narrows what is read through that dataset, not a statement that reads the tables beneath
  it.
- **`RLS_IN_SQLLAB`.** At its default, off, SQL Lab's own screen applies no row rules. A builder holds the sql_lab
  role, so in SQL Lab they read the rows the rules keep out of their answers through mnemiq. Superset's MCP service
  applies row rules to every statement whatever the flag says. Turning it on is the deployment's call; Superset's own
  configuration marks the flag as in testing.

## mnemiq's settings

| Setting | Value |
|---|---|
| `MNEMIQ_SUPERSET_MCP_URL` | Superset's MCP service, which `superset mcp run --port 5008` serves at `http://<host>:5008/mcp`. It listens on `127.0.0.1` only unless `--host` names another address, so mnemiq on another host needs `--host`. Setting it turns Superset mode on. |
| `MNEMIQ_SUPERSET_DATABASE_ID` | The id Superset gives the database mnemiq answers from. |
| `MNEMIQ_SUPERSET_SCHEMA` | The schema mnemiq reads in that database, its name spelled as PostgreSQL stores it; `public` by default. |
| `MNEMIQ_OIDC_ISSUER`, `MNEMIQ_OIDC_AUDIENCE` | Your provider's issuer, and the audience a builder's token carries. Superset's `MCP_JWT_ISSUER` and `MCP_JWT_AUDIENCE` hold the same values. |
| `MNEMIQ_TENANT` | The tenant every token must carry. |
| `MNEMIQ_SUPERSET_SERVICE_KEY_FILE` | For `mnemiq enrich` and `mnemiq refresh` only: the file holding the service user's API key. Not a server setting. |

mnemiq needs no database credential: it opens no connection to the source in this mode. An http
`MNEMIQ_SUPERSET_MCP_URL` to a host other than this machine starts with a warning, since each builder's token then
crosses the network unencrypted. That suits a private network, such as a Compose stack's own; anywhere else, use
https. [Settings](settings.md#the-data-source) has the rules each one is held to.

## The service key

Enrichment builds the snapshot offline, before anyone asks anything, so it reads the source as a Superset user of its
own: the service user.

1. **Create the service user** in Superset, with the roles above.
2. **Create an API key for it**, with API keys on. A key made with scopes is held to them as a token is, so it needs
   `superset:sqllab:write` and `superset:database:read`; a key with none is held to the user's roles alone.
3. **Write the key, alone, to a file only its owner can read,** and name the file in
   `MNEMIQ_SUPERSET_SERVICE_KEY_FILE`:

   ```bash
   chmod 600 "$HOME/.config/mnemiq/superset-service.key"
   ```

   The file must be a regular file that grants its group and others nothing (`0600`; `0400` passes too), at most
   64 KiB, holding one key: visible ASCII with no spaces, once surrounding whitespace is stripped. Anything else is
   refused with a sentence saying why, which never holds the key.

The key is not a server setting. `mnemiq enrich` and `mnemiq refresh` read it, only the command line imports the
module that reads it, and a test holds every module to that, so the server, which answers with each builder's own
token, never holds it. Set the variable where enrichment runs, not in the server's environment.

## Enriching

```bash
MNEMIQ_SUPERSET_SERVICE_KEY_FILE="$HOME/.config/mnemiq/superset-service.key" mnemiq enrich
mnemiq build
```

- **PostgreSQL only.** Before it reads anything, enrichment asks Superset (`get_database_info`) which backend the
  database is, and refuses anything but PostgreSQL.
- **No values.** It reads the catalogue and the counts, and harvests no column values: no samples, no value index, no
  codes read from the data, and no binding suggestions. Explicit concept bindings in your ontology records still
  apply, since they read nothing from the source. It clears the value index and the stored examples for its source,
  so nothing an earlier direct run left behind is served.
- **One exception, partition labels.** For a column with at most two distinct values in a table whose other columns
  repeat, it records those two values as partition labels. The fan-out guard reads them to know which keys are unique
  within each partition, and they are never shown or sent to the model.
- **The source as it is now.** It asks Superset to run every statement afresh, past Superset's result cache, and
  gives each statement up to 300 s.
- **Where it read from.** The snapshot records the database and schema it was read from, which the server checks
  before it serves the snapshot.
- **Experimental phases refused.** `MNEMIQ_ENRICH_FACTS` and `MNEMIQ_ENRICH_EXAMPLES` must be off.

`mnemiq refresh` reads the catalogue through Superset the same way, and enriches and builds again when the catalogue
changed. It also does so when the stored snapshot is one this deployment would not serve, even with the catalogue
unchanged.

## Serving

```bash
mnemiq serve --http
```

`serve --http` checks the Superset settings before it builds anything, and refuses to start, saying what to change,
when:

- `MNEMIQ_SUPERSET_DATABASE_ID` is missing; or it is set without `MNEMIQ_SUPERSET_MCP_URL`, which is refused in direct
  mode too, since that deployment would otherwise read the source with its own credential;
- `MNEMIQ_SUPERSET_MCP_URL` is not plainly an http or https URL naming a host, or carries a user name, a password, a
  query or a fragment;
- `MNEMIQ_SUPERSET_SCHEMA` is not letters, digits and `_`, starting with a letter or `_`;
- `MNEMIQ_OIDC_ISSUER` is unset;
- `MNEMIQ_MCP_TRANSPORT=http`: MCP over HTTP is not served in Superset mode;
- `MNEMIQ_OPENWEBUI_JWT_SECRET` or `MNEMIQ_OPENWEBUI_AUDIENCE` is set: neither door's tokens are meant for Superset;
- `MNEMIQ_WRITE_ENABLED` is on;
- `MNEMIQ_SOURCES_PATH` is set;
- `MNEMIQ_AUTHZ_PATH` is set: Superset's roles and row rules decide what each builder may read, and mnemiq applies no
  policy file of its own there;
- `MNEMIQ_ENRICH_FACTS` or `MNEMIQ_ENRICH_EXAMPLES` is on.

As it builds the runtime, it also refuses to start when:

- `MNEMIQ_MODE=deep`: `deep` is not offered. A request asking for it is refused: `/v1/ask` answers 422, naming the
  modes this deployment offers;
- the snapshot was not enriched through Superset from the configured database and schema, or its enrichment did not
  finish. The message says which, and what makes one it serves: `mnemiq enrich` in Superset mode, then `mnemiq build`.

With a control database (`MNEMIQ_CONTROL_DSN`), a newer snapshot published later is held to the same rule. If this
deployment would not serve it, every question and every `/v1/schema` request is answered 503 until one it serves is
published; its log says why, and `mnemiq refresh` enriches it again.

**Commands that answer as one identity refuse Superset mode:** `mnemiq ask` and MCP over stdio (`mnemiq serve`
without `--http`) name `mnemiq serve --http` instead, and `mnemiq write` says Superset mode is read-only. Anything else
that would open mnemiq's own connection to the source, `mnemiq eval` among them, refuses too.

## What each refusal returns

A request Superset refuses, or that Superset mode cannot serve, is answered with a status and one sentence saying what
the builder can do about it, or that nothing they can do is known to help. None of these is a 500.

| What happened | Status | The sentence |
|---|---|---|
| Superset does not accept the token, or has no user for it | 401 | "Superset did not accept this sign-in; sign in to Superset again." |
| The builder's roles, or the token's scopes, do not permit a tool mnemiq calls | 403 | "Superset refused this user's queries. Building with mnemiq needs Superset's Alpha and sql_lab roles." |
| Superset refuses and does not say why | 403 | "Superset refused this request and did not say why; its log records the reason." |
| Superset's MCP service answers with an error that is not a refusal, while mnemiq asks what the builder may read, before the answer starts | 503 | "Superset is not set up as mnemiq needs: " and what failed, for example "Superset's MCP service does not offer execute_sql, which Superset mode calls, and which Superset 7.0.0rc2 and later offer unless MCP_DISABLED_TOOLS names it" |
| Superset does not answer in time, or answers in a way mnemiq cannot read, while mnemiq asks what the builder may read. That is the first thing mnemiq asks Superset for a question or a `/v1/schema` request, so an MCP service that is down or unreachable is answered this way | 503 | "Superset's answer about what this user may read could not be read, so nothing was answered; mnemiq's log records why." |
| A newer snapshot was not enriched from this database and schema | 503 | "This deployment's knowledge is not the one enriched for the Superset database it answers from, so nothing is answered until an operator enriches it through Superset; mnemiq's log records why." |
| A request with no token | 401 | the door's own refusal, as on every `/v1` route, before Superset is asked anything |

- **Once the answer has started, an outage is not a 503.** A tool error is the model's to repair, like a rejected
  statement. Superset not answering ends the answer inside a 200, marked `failed` with `reason_code`
  `source_unavailable`, and a sentence saying the source did not answer, that this is not about the query, and to try
  again. A client that watches for outages reads the answer's `reason_code` as well as the status.
- **Nothing writes.** No `/v1` route writes, and `mnemiq write` refuses Superset mode.
- **What Superset said goes to mnemiq's log only.** Its text can name a table, a role or a user the builder may not
  see.
- **A 401 asks for the Bearer again** (`WWW-Authenticate: Bearer`), and a 403's body carries `end_session_url`, as
  every `/v1` 403 does, so the workbench can let the builder sign out and switch accounts.
- **On `/v1/chat`,** whose 200 has already gone out, the stream's error frame carries the same sentence.
- **A refusal partway through an answer ends it as that refusal**, wherever it is met: a token Superset stops
  accepting, roles that do not permit a tool, or a denial Superset does not explain. It is never answered with a
  repair or "try again".
- **A statement Superset refuses because it reads something the builder may not read** is different: it is the
  model's mistake. The model repairs it from mnemiq's own sentence, and Superset checks the repaired statement again.

## Capacity

- **One slow statement delays every call on its MCP process.** Superset's MCP service runs a statement on its event
  loop, so every other call to that process waits behind it. Measured: a `SELECT 1` waited 5.75 s behind a 6.6 s
  statement. Run several MCP service processes, and set `statement_timeout` (above) so no statement holds a process
  for long. mnemiq takes one `MNEMIQ_SUPERSET_MCP_URL`, so the processes need a load balancer in front: with
  Superset's default stateless HTTP (`MCP_STATELESS_HTTP = True`) a plain one works, and with it off the balancer must
  route each session by `Mcp-Session-Id` to the process that created it.
- **Each question makes several calls.** mnemiq opens an MCP session for each call. A question costs the dry run for
  its grants, the catalogue reads for the function inventory, which are read afresh for every request, the statement's
  proof and the statement itself, and more when the model repairs one. Measured: twelve calls for one question on the
  contract suite's small database.

## What Superset mode does not do

- **No sample values.** A snapshot enriched through Superset holds no column values, and the server serves no value
  index and no stored examples, whatever the store holds. A value a question names is checked against the source only
  by the statements that run as the builder, never against values stored ahead of time.
- **No policy file and no column masks of mnemiq's own.** Superset's roles and row rules govern, and Superset has no
  column rules for SQL.
- **No asking with access to datasets alone** (see [Roles in Superset](#roles-in-superset)).
- **A row rule attached only to a virtual dataset** applies through that dataset, not to the tables beneath it (see
  [Row rules](#row-rules)).
- **No writes, no `deep` mode, no MCP door and no Open WebUI endpoint.**
