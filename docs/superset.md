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
  limit. mnemiq never sends `template_params`, and sends `limit` only to ask again for an answer Superset refused as
  larger than one response, below the statement's own LIMIT (see "Results that fit one response" below). A statement
  holding `{{`, `{%` or `{#` is never sent,
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
- **Results that fit one response.** Superset's response size guard (`MCP_RESPONSE_SIZE_CONFIG`, 50,000 bytes by
  default) cuts a larger `execute_sql` result by dropping rows, or refuses it when no cut fits. An answer to a question
  is then the result's first rows: those Superset kept, or, when it refused, those a smaller `limit` brings under the
  guard, asked again up to three times, each sized from Superset's refusal. The answer says so: "These results stop at
  the N rows Superset sends in one answer, so there may be more: counts and totals over them are not the whole."
  Nothing else mnemiq reads, a chart's check, the catalogue or a plan's probe, is read from a cut result. For whole
  results, take `execute_sql` out of the guard, which is the better fix, or raise its `max_bytes` to about twice the
  size of the largest result you want answered whole: the guard measures the whole answer, and `execute_sql`'s answer
  carries every row twice, once as the result's rows and again in the statement's own data.
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
- **So a chart on a virtual dataset is never checked against mnemiq's own SQL, with one exception:** a dataset the
  build itself made, whose SQL is still the SQL the build made it with, and on which no row rule applies to the builder.
  The extension's `row_rules` asks Superset whether one applies, as the builder, and answers only whether. When one
  does, or mnemiq cannot tell (an extension older than the tool, say), mnemiq's statement is not run and the chart is
  not checked. A saved metric on any other virtual dataset is checked through Superset's own query of that dataset.
- **`RLS_IN_SQLLAB`.** At its default, off, SQL Lab's own screen applies no row rules. A builder holds the sql_lab
  role, so in SQL Lab they read the rows the rules keep out of their answers through mnemiq. Superset's MCP service
  applies row rules to every statement whatever the flag says. Turning it on is the deployment's call; Superset's own
  configuration marks the flag as in testing.

## mnemiq's settings

| Setting | Value |
|---|---|
| `MNEMIQ_SUPERSET_MCP_URL` | Superset's MCP service, which `superset mcp run --port 5008` serves at `http://<host>:5008/mcp`. It listens on `127.0.0.1` only unless `--host` names another address, so mnemiq on another host needs `--host`. Setting it turns Superset mode on. |
| `MNEMIQ_SUPERSET_DATABASE_ID` | The id Superset gives the database mnemiq answers from. |
| `MNEMIQ_SUPERSET_SCHEMA` | The schema mnemiq reads in that database, its name as Superset shows it: on PostgreSQL as stored, on Oracle a name stored in upper case in lower case (`SHOP` as `shop`); `public` by default. |
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

## Building

With the mnemiq extension installed in Superset ([`integrations/superset`](../integrations/superset/README.md)), a
builder can turn a goal into a dashboard. mnemiq plans the charts on named measures, makes them in Superset as that
builder, lays them out, and checks each chart it can against its own SQL, run as the same builder. A fuller guide is
still to come. In short:

- **The build routes, mounted only in Superset mode.**
  - `POST /v1/superset/builds` takes `{goal, dashboard_id?, fresh?, profile?}`. It starts a build. With neither
    `dashboard_id` nor `fresh`, it may instead answer an offer: a dashboard the builder can see that already covers
    the goal. An offer builds and keeps nothing.
    - To reuse the dashboard offered, open it: that needs no build.
    - To add this goal's charts to it anyway, ask again with its `dashboard_id`. A `dashboard_id` always means that:
      the goal's planned charts are added to the dashboard it names, beside the ones it has.
    - For a new dashboard, ask again with `"fresh": true`.
  - Onto a dashboard, mnemiq first reads the charts it holds, as the builder, and leaves out each planned chart one of
    them already shows: the same type and dataset, the same measures by definition, the same grouping and x-axis
    columns, time grain, filters and row limit. The answer names each one left out and the chart that shows it
    (`left_out`). A goal the dashboard shows whole builds nothing and is answered as `shown`. A chart mnemiq cannot
    read the same way, a filter written in SQL among them, is never taken for a match, so the planned one is built.
  - Onto a dashboard, the plan is also shown the measures the dashboard keeps that mnemiq would run for the builder: one
    on a table's own dataset the builder can read, or one whose statement over its dataset, as Superset answers the
    dataset to the builder, mnemiq's deciders approve for them. The others reach the model in no part. A dashboard keeps
    one definition for each measure name, and a measure's dataset is part of its definition: the same formula on a
    dataset this build makes is a measure of its own. So a planned measure that takes a kept name with another
    definition, or a kept label under another name, goes back to the model once for a name and label of its own (in
    words that say nothing of a measure the builder may not be told of), and the plan is refused if it still does. The
    measure the dashboard keeps, planned again as it is (its kind, dataset and expression), is that measure, and the new
    charts join it.
  - Before anything is planned, mnemiq asks Superset whether the person may build at all; one who may not is
    refused with a 403 as a non-builder, and nothing about a dashboard they named is read. Then, with a
    `dashboard_id`, it asks whether they edit that dashboard. One they cannot read, or one that does not exist, is a
    404 that does not say which. One they can read but not edit is a 403 naming its editors.
  - A model that does not answer the planning prompt is a 503 saying to try again: an outage, told apart from a goal
    that cannot be planned by its status. A model server that refuses or cuts the planning prompt as longer than its
    context window gives a refused plan (a 200): the goal and what the builder can see are too large to plan in one
    prompt. The planning prompt holds every table, column and saved metric the builder can see, so a large schema
    needs a large window.
  - `POST /v1/superset/builds/{id}/next` runs the build's next step.
  - With `"propose": true`, the start plans the build and keeps it as a proposal, writing nothing to Superset: a 200
    naming the build, its goal, its dashboard and its plan. `POST /v1/superset/builds/{id}/confirm` starts it,
    answered as `/next` answers a build, and `/next` then runs it as any build. A proposal holds no dashboard, so
    another build of the dashboard may run meanwhile. At the confirm, a proposal onto a dashboard is refused with a 409
    while another build of the dashboard is building, and when the dashboard changed since the plan (its `changed_on`,
    read before planning): plan it again. A second confirm is a 409 too.
  - `GET /v1/superset/builds/{id}` reads the build as it stands. Every build answer names its plan (each chart's
    title, family and measures, and each measure mnemiq writes), and the check step's own answer says why each chart
    that did not check did not (`check_reasons`), which is not kept.
  - `GET /v1/superset/builds` lists the person's builds still building, newest first (with `?dashboard_id=`, only
    that dashboard's), each with its goal and how many of its steps are done: what a client offers to finish.
- **A dashboard's measures**, for any editor of the dashboard, each route asked with a token minted for building:
  the list reads the dashboard's layout and charts, which a token minted for asking cannot (a 403 naming the scopes).
  - `GET /v1/superset/dashboards/{id}/measures` lists the measures mnemiq keeps for the dashboard: each definition,
    the charts on the dashboard that show it (by id and title, with each one's last check), and its value, computed
    then as the person asking and kept nowhere. On a physical dataset the value is mnemiq's own statement, run through
    Superset as that person, so Superset's row rules apply to it. On a virtual dataset it comes only through Superset:
    a saved metric by Superset's query of the dataset, a written measure by a chart that shows it as one number.
    Otherwise there is no value, and `value_note` says why. `value_basis` says what a value covers:
    `readable_rows`, every row of the measure's dataset the person can read, before any chart's filters; or `chart`,
    the one number the chart `value_chart_id` shows, under that chart's own filters. Both are null when there is no
    value.
  - `POST .../measures/{name}/edit` takes `{label, expression, kind, currency?, description?}`. The new definition
    passes mnemiq's measure checks first, and its label may be no other measure's on the dashboard. Every copy on a
    chart that still matches the kept definition changes, and each changed chart is checked again; a copy changed by
    hand is left and named. When a chart keeps changing under the edit, the charts already changed are put back and
    the dashboard keeps the old definition, so its copies never differ.
  - `POST .../measures/{name}/remove` takes `{chart_ids}`, the charts the person was shown, and answers 409 with the
    charts the measure has now when they differ. It moves the charts that still show the measure to Superset's trash
    (or deletes them for good where Superset keeps none, and says so), leaves a chart changed by hand and names it,
    takes a chart the person does not edit off the dashboard instead and names it with its dataset and editors,
    names a chart a report holds and leaves it, and deletes a dataset mnemiq made for the measure once no chart uses
    it. A chart already in Superset's trash is no longer listed, and is taken off the layout as gone.
  - `POST .../measures/{name}/save` adds the measure to its dataset as a saved metric, then switches every matching
    copy to it. Asked again, it switches what is left onto a saved metric of the measure's own definition; one of the
    name with another definition refuses it.
  - Each action first asks whether the person edits every chart it would change (and, to save, the dataset). Edit
    and Save change nothing unless they edit every chart that shows the measure, and name that chart's editors, so
    the copies never split. No action runs while a build is building on the dashboard (409). Each answers
    `changed`, `left_by_hand`, `not_editor` (each chart with its dataset, or a dataset, and the editors),
    `blocked_by_report`, `could_not_read` and mnemiq's `message`.
- **Changing a dashboard by chat**, for any editor of it. `POST /v1/superset/dashboards/{id}/changes` takes
  `{request}`. The profile's model reads the request against the dashboard's charts as the person sees them (each
  chart's title, type, table, measures by label, grouping, grain, row limit, filters, and the columns of its table
  they may read; never an editor or a value from the data). mnemiq then writes only the settings the request names,
  through the extension's `patch_chart`, which leaves every other setting, a hand edit among them, as it is.
  - Only the settings a chart's type may have changed by chat are written, each with a value mnemiq reads as valid;
    any other is dropped and named. A chart's type changes only when the request asks, and only among line, bar,
    area and scatter. Filters are added to a chart's own, or dropped by column: the chart's other filters stay.
  - A chart changed meanwhile is read again and changed once more; changed again, it is left and named. A chart the
    person does not edit is named. A filter the chart already has is not added twice.
  - Nothing is deleted on the model's reading: a chart the request asks to remove is answered under `would_remove`,
    by id and title. Sent back as `{remove_chart_ids}`, once the person confirms, those charts go to Superset's trash
    and off the layout (and the model is not asked); one a report holds is left, with the reason.
  - A request for a chart the dashboard lacks is planned onto it and started as a build (`added`, as the start
    route answers it), which the client runs step by step. When that planning fails, what the request changed is
    still answered, and `added` holds the failure's status and sentence.
- **Values a plan's repair is told.** The planning prompt holds no value read from the data's rows. When a plan
  filters on a value no row the builder can read holds, up to 20 values of that column, one the plan itself filtered,
  read as the builder, go to the one repair prompt and to the refused plan's answer: a hosted planning model sees them.
  A named value that differs from the plan's only in letter case is called out, and the advice is to filter on one of
  the values only if it is the one the goal means, and to keep the plan's own value only if the goal itself names it.
- **A negated filter on a column that holds NULL.** SQL reads a chart filter with `!=`, `NOT IN` or `NOT LIKE` as
  leaving out every row where its column is NULL. When a row the builder can read holds NULL in that column, the chart
  is refused, repairably, and the repair is told the three ways out: the filter replaced with `IS NULL` if only those
  rows are meant, `IS NOT NULL` beside the filter if leaving them out is meant, or a filter on another column. A chart
  that already filters the column `IS NOT NULL` is not asked. One that filters it `IS NULL` beside the negation selects
  no row, since a chart's filters all apply, and is refused without asking. The probe runs as the builder and says
  only whether such a row exists: no value reaches the model.
- **Columns named with their tables, and the rows a measure reads.** The planning model names every column as
  `<table>.<column>`, in a measure's expression and in a chart's columns (`orders.order_date`,
  `SUM(order_items.revenue)`), and never writes a `table` or a join. mnemiq reads the rest from those names.
  - **A measure's base** is the table of the columns inside its aggregate. A measure that adds up no column (a count or
    sum of 1s, a share written from 1.0 and 0) counts the rows of the table its condition names. It gives `table`, the
    one table whose rows it counts, where its aggregate names no column (`COUNT(*)`), where its conditions name columns
    of two tables, or where the rows it counts are another table's (loans counted by their book's format). A `table`
    given decides over the condition; a measure that adds up a column is that column's table's, whatever its `table`.
  - **Links.** Where the known links (the snapshot's relationships, which the prompt shows) do not connect a table a
    measure or chart needs, the model states one: `{"from": "order_items.order_id", "to": "orders.order_id"}`, from the
    table whose rows reference to the table referenced. A stated link decides its pair of tables.
  - **One dataset per base.** When a base needs nothing else, its measures and charts use the table's own dataset.
    When a chart or measure needs another table, mnemiq makes a virtual dataset: the base `LEFT JOIN` each needed table,
    along the shortest chain of links whose columns are of one kind and whose far side holds each value once. A
    `LEFT JOIN` from the base keeps every row of the base, so a measure over it counts the base's rows as it would
    alone. Among equally short chains, the one
    with the most stated links wins. Two chains with none preferred, two different links between one pair of tables,
    or no chain at all go back to the model once for the repair, which is told to state the link.
  - **A link joins two columns of one kind.** An equality of two columns of different types converts one side, and a
    conversion that rounds reads two values as one: a `double precision` equal to a `bigint` compares both as
    `double precision`, which reads two ids past 2^53 as one, so a key that holds each value once can still meet a row
    twice. mnemiq joins by a link only where both columns are integers or exact numerics (`smallint`, `integer`,
    `bigint`, `numeric`), both `text` or `varchar`, or both of one other type (`char` with `char`, `date` with `date`,
    a timestamp with a timestamp of the same time zone kind), lengths and precisions aside. A link of two other kinds,
    or with a column whose type the snapshot does not record, is never joined by, whether the model stated it or the
    snapshot knows it, and nothing is read of its key: the repair is told both columns and both types. Every equality
    of a made dataset's `LEFT JOIN` is held to the same rule, whoever wrote the join.
  - **A far side that repeats is never joined,** since it would repeat the base's rows. Enrichment's profile says
    whether a key holds each value once. Where it does not say either way, mnemiq measures it as the builder, through
    Superset: `COUNT(*)` against `COUNT(DISTINCT ...)` of the key's non-null values, and joins only if the table holds
    each value once. Superset applies its row rules to that count, so under a rule it would count only the builder's
    rows, and a key unique there may repeat in rows another person reads. So before it counts, mnemiq asks the
    extension's `table_rules`, as the builder, about the table by name on the deployment's database and schema:
    Superset's own lookup of the rules a statement reading the table takes, through whichever of the table's datasets
    it finds. Where a rule applies to the builder, or the answer cannot be read (an extension older than 0.3.0, which
    has no `table_rules`, among such answers), it does not count, and the join is refused. So is a join on a key
    column the builder's grants mask or deny. The repair is told why.
  - **Column names on the made dataset** follow one rule, which the model never writes: a column only one included
    table has keeps its name, and one that two share is `<table>_<column>` (`orders_region`, `customers_region`).
  - **Measures are decided over the made dataset** exactly as Superset reads it, as `(<its SELECT>) AS virtual_table`,
    under the builder's grants, as every measure is.
  - Every measure on a base lives on that base's one dataset. A certified saved metric of a base whose charts need
    another table is therefore carried onto the made dataset as a copy, with the same expression and so the same
    numbers, and those charts do not show Superset's certification.
  - Building onto a dashboard, the prompt shows the dashboard's kept measures by name and label only, and the model
    gives a new measure a new name and label.
- **A saved metric on a made dataset.** A plan may name a saved metric of a table instead of writing its own measure,
  and where that table's charts need another table, mnemiq carries the metric onto the made dataset as its table
  defines it. It prints the metric's expression twice from one parse: once with each column named by its table's
  place in the join, which it decides under the builder's grants as it decides every measure, and once naming each
  column as the made dataset does (`<table>_<column>` for a column two tables share), which the chart sends as its SQL
  metric, labelled with the metric's verbose name. Its values are its table's, so none is probed. It is refused, and
  the repair told why, where it would not mean what its table defines:
  - its expression reads anything but its own table's columns;
  - its expression names none of its table's columns (`count`, `COUNT(*)`), so it counts the rows it is given, and the
    join's rows are not its table's. A dataset mnemiq makes has exactly its base's rows, each once, when every far
    side is shown to hold each value once, so the base's `count` is carried onto it as written; on any other join it
    would count the join's rows. A base with no dataset is offered the `count` Superset gives the dataset mnemiq makes
    for it; on such a made dataset its chart plots the made dataset's own `count`, the same count;
  - its table is on a side an outer join may fill with NULL: the right of a `LEFT JOIN`, and, in a join mnemiq did not
    make (it makes only `LEFT JOIN`s), every table before a `RIGHT JOIN` and any table of a `FULL JOIN`;
  - with the fan-out guard on (`MNEMIQ_GUARD_FANOUT`), the join repeats its table's rows, meets another table by a
    key not known to be unique there (one with no uniqueness profile, say), or joins it by no key to every other
    table, whatever the metric aggregates.

  Its chart, like every chart on a dataset a build makes over a join, is checked against mnemiq's own SQL over the
  dataset's SQL, read as Superset reads a virtual dataset, while that SQL is unchanged and no row rule applies to the
  builder on the dataset (see [Row rules](#row-rules)).
- **A token made for building.** Besides `db.read`, the two `POST`s need Superset's `superset:chart:write`,
  `superset:dashboard:write` and `superset:dataset:write` scopes. The extension's token route puts those in a token
  asked for with `{"purpose": "build"}`. A token made for asking is refused with a 403.
- **One step per request.** Each `/next` claims one step, runs it as the builder and records what it made, so a build
  survives a restart. A step holds a ten-minute lease while it runs, so two requests never run the same step: a
  second `/next` meanwhile is refused with a 409.
  - Every dashboard, dataset and chart a build makes is made by the extension, with the build's marker in its
    description, saved in the same transaction as the object. Sent again with the same marker, a create answers with
    the object the first one made instead of making another.
  - So if a create times out, the next `/next` sends it again at once. If Superset made the object, mnemiq gets its
    id back; if not, it is made now. If Superset is still running the first create, the second waits for it to finish
    and answers with what it made; if it waits too long, or Superset's metadata database does not answer, the next
    `/next` asks again.
  - Only an object that carries the step's marker is ever taken as the step's own. A chart someone makes by hand with
    the planned name is left alone. If two objects carry the same marker, the build stops and names those the
    builder can read.
- **One build of a dashboard at a time.** While a build of a dashboard is building, starting another build of it is
  refused with a 409. A build left untouched for 20 minutes (twice the lease), with no step running, no longer holds
  its dashboard: the next build of that dashboard stops it, as abandoned, and takes the dashboard over.
- **A chart whose row limit keeps only some of its groups is checked as a ranking** when it is a bar, line, area or
  scatter chart, a table, a pie, a gauge or a treemap. The check reads from the chart's saved settings which measure
  its query ranks by, and in which direction, as Superset orders it. It then asks whether the groups shown are the ones
  that ranking puts first among all of mnemiq's own groups. A group with no value for that measure counts after every
  group with one, and groups tied at the cut may stand in for each other. Any other chart a row limit cuts is not
  checked: a mixed chart, a pivot table, a funnel, a sankey or a waterfall, or one sorted by a column.
- **A chart the builder makes with a row limit leaves out the groups its measure has no value for**, when it is one of
  those types and is not over a time axis. Its query gets `HAVING (<measure>) IS NOT NULL`, so the groups the limit
  keeps are the largest with a value, and the check leaves the same groups out of its own rows. A chart over a time
  axis gets no such HAVING: there it would drop points from a series and join the line across them.
- **Builds are kept in the control Postgres (`MNEMIQ_CONTROL_DSN`).** It holds plans, measure definitions, Superset
  ids and statuses, never a value or a row. Without it the build routes answer 503.
- **Without the extension, the build routes answer 503** with a sentence that names it, before anything is planned. An
  extension older than mnemiq, without the creates a build sends, is answered 503 too, with a sentence saying to
  update it. One that has them but does not take a field a chart needs (`create_chart`'s `leave_out_no_value`, for a
  chart made with a row limit) stops the build at that chart's step, with a sentence saying to update it.

## Conversations

A person's conversations with mnemiq in the panel are kept in the control Postgres beside the builds
(`mnemiq_superset_conversation`, made the first time the store connects), each its tenant's and person's alone: another
person's conversation id is a 404 on every route, and says nothing of it. The panel's forwarder mints each request a
token for building; mnemiq asks of it `db.read` for every route, and Superset's write scopes for a build or a change.
`MNEMIQ_CONTROL_DSN` unset answers each route with a 503, as the build routes do.

- `GET /v1/superset/conversations` lists the person's conversations, newest first (`limit`, at most 100; `before`,
  the `next` cursor of the page before). `POST /v1/superset/conversations` starts one; its title is set from its first
  message. `GET .../{id}` reads one with its messages. `POST .../{id}/delete` deletes it, withdrawing the builds it
  proposed that nobody confirmed; a build it started keeps its own record.
- `POST .../{id}/messages` sends a message, with the dashboard on screen (`dashboard_id`, its title and whether it has
  charts, as the panel shows them), and answers an event stream. The profile's model first reads what the message
  asks, in one small call (`mnemiq.routed`): a question about the data, charts for the dashboard on screen, a new
  dashboard, a change of the dashboard on screen, or that it cannot tell. Then:
  - **a question** streams exactly as `/v1/chat` streams it, the conversation's earlier answers as its history;
  - **a build** is planned and kept as a proposal that writes nothing to Superset (`mnemiq.proposal`), or answered
    as an offer of a dashboard that covers it, a refused plan, or a dashboard that already shows it all;
  - **a change** is previewed with nothing written (`mnemiq.preview`): each chart's edit, and the charts it would
    remove;
  - **a message it cannot read**, or one that asks for a change or for charts off a dashboard, is a question back
    with three choices (`mnemiq.clarify`); `choose` sends the message again where the person chose.
  A build or a change reaches the planner or the change reader rewritten to stand alone ("add that as a chart here"
  names what "that" was), and the person sees it in the plan or the preview. A question reaches the Ask path exactly
  as typed.
- `POST .../{id}/messages/{n}/confirm` is the person's "Build it" or "Apply", sent from the dashboard the proposal is
  for (a 409 from anywhere else, the proposal kept). The message is claimed before anything is written, so a second
  confirm, from a double click or another tab, is a 409 and does nothing.
  - A build is confirmed as the build route confirms it; on a dashboard another build is building, or one changed
    since the plan, the proposal is withdrawn and the goal planned again, as a new message that waits in turn.
  - A change is written exactly as previewed, with no second reading by the model. Nothing is written when the
    dashboard or a chart the preview names changed since it was read; a chart changed by hand between the check
    and its write stops the rest, what was written named.
  - A removal the preview proposed is done only for the charts the confirm names (`remove_chart_ids`), and only
    once every edit beside it is written: a chart to remove that changed since the preview, like a stale edit,
    leaves the whole preview unwritten.
  - A confirm that fails before anything is written leaves the proposal waiting; one that fails partway is recorded
    as done in part, naming what was written, and is not applied again.
  - An offered dashboard takes `choice`: `add` proposes this goal's charts onto it, `new` a new dashboard.
- `POST .../{id}/messages/{n}/decline` is "Not now": the proposal is set aside, and a build's withdrawn.
- `POST .../{id}/messages/{n}/results` runs a kept answer's SQL again, decided as a new question's SQL is and run as
  the person, now: its rows are answered and kept nowhere. SQL the person can no longer run is answered with the Ask
  path's own sentence. An answer mnemiq declined or could not give has nothing to run again (404).

A conversation keeps each question, mnemiq's sentence for it and its SQL, a build's plan as the person was shown it,
and a change's preview (a filter's typed value among it). An answer mnemiq declined or could not give keeps that too,
with its reason's code, and is no follow-up's history. It keeps no row of results: every message is held to an
allow-list of its keys at every depth, and any key a value, a row or a sample is kept under is refused.

## The panel

The extension ([`integrations/superset`](../integrations/superset/README.md)) also puts mnemiq inside Superset 7.0 as
the page's chat. A trigger in the page's corner opens a panel that floats over the page or docks beside it, for a
signed-in person, with `ENABLE_EXTENSIONS` on. The panel shares Superset's React and antd, and reaches mnemiq only
through the extension's forwarder, as the person signed in: the browser holds no mnemiq token and never learns
mnemiq's address.

- **Who sees what.** Each time the panel opens it asks mnemiq once whether this person may use it. A person Superset
  will not let build (Alpha and sql_lab are needed) sees only the sentence that says so, and the panel asks nothing
  else.
- **One box.** The person types; mnemiq works out whether it is a question, a dashboard to build, or a change to the
  dashboard on screen (see [Conversations](#conversations)), and asks back, with three choices, when it cannot tell.
- **Their conversations**, kept by mnemiq: Conversations lists them, newest first, and opens one, or deletes one after
  a second click; New starts another, which mnemiq makes with its first message. Each message shows the dashboard it
  was sent from when that changed. While a message, a Build it or an Apply is in flight, the panel stays on its
  conversation.
- **Ask** a question about the data. The answer streams in: its sentence, the SQL (folded), and the rows it read. Reopened
  later, an answer shows its sentence as it was written then and its SQL, and Show results runs that SQL again as the
  person, now; the rows are kept only while the card is open. One mnemiq declined or could not give reopens as it
  read then: its reason and what to try next, and no Show results.
- **Build** a dashboard. mnemiq shows the plan and makes nothing until Build it; then the panel runs the build one step
  per request, with each chart's check, and links to the result. A build onto a dashboard goes onto the one on screen,
  and on a dashboard with no charts the box opens with "Describe what this dashboard should show". Off a dashboard,
  mnemiq may first offer one that already covers the goal, with Open it, Add these charts to it, and Build a new one.
- **The dashboard's measures**, on a dashboard the person can read: each measure mnemiq keeps there, with its
  definition and its value computed now under the person's own row rules, and Edit, Remove and Save to dataset.
  Remove deletes the charts that show the measure (into Superset's trash, where it keeps one), and takes a chart the
  person does not edit off the dashboard instead; it says so, and names the charts, before it does anything.
- **Change** the dashboard on screen: "show daily instead of monthly". mnemiq shows each chart's edit, what it will not
  change, and the charts the change would delete, each with its own tick. Apply writes exactly what it showed, and
  nothing when the dashboard or a chart changed since; a chart changed by hand while it writes stops it there, and the
  card says what was changed and what was left. A chart setting changed by hand is kept. Build it and Apply are
  offered only on the dashboard they are for, read from the address when clicked, Build it not while another build
  runs on the page, and Not now sets a proposal aside.
- **Finish.** Closing the panel does not stop a build. Leaving or reloading the page stops it after the step in
  flight, and the build stays resumable: the panel then lists the person's unfinished builds (on a dashboard, that
  dashboard's), each with Finish and Not now.
- **What it keeps: nothing of its own.** mnemiq keeps the conversations; the panel holds what is on screen, and a
  running build, in the page's memory for its life. A reopened answer's rows and a measure's value live only in their
  open card or section. Nothing is written to browser storage.
- **Reload to see the changes.** A finished build or change offers it, so Superset draws what was written. It carries
  the conversation on screen to the reloaded page in the address's fragment (`#mnemiq-conversation=<id>`, which the
  browser never sends to a server), and the panel reopens that conversation from mnemiq when it next opens, then
  clears the fragment. A reload of the browser's own starts the panel empty, as before.

The forwarder is configured in Superset's `superset_config.py`, beside the token settings the extension's README lists
(it mints each request's token with them):

| Setting | What it is |
|---|---|
| `MNEMIQ_URL` | Where Superset's web server reaches mnemiq. Required. |
| `MNEMIQ_FORWARD_TIMEOUT_S` | How long a forwarded request may take, its whole answer read, 1 to 900 seconds. Default 600. |
| `MNEMIQ_FORWARD_IDLE_S` | How long a chat answer's stream may go quiet between two reads, 5 to 300 seconds. Default 60. |

Planning a build with a slow model can take minutes, and each forwarded request holds one of Superset's web worker
threads while it runs. A proxy in front of Superset must allow a read timeout at least as long as
`MNEMIQ_FORWARD_TIMEOUT_S`, and must pass a chat answer's event stream through unbuffered; size Superset's workers for
the builds that run at once. The README lists every refusal the forwarder answers.

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
| The same, on a deployment that answers from Oracle | 503 | "This deployment's knowledge is not the one enriched for the Oracle database it answers from, so nothing is answered until an operator enriches it directly from Oracle for Superset mode (`mnemiq enrich --for-superset`); mnemiq's log records why." |
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
- **A row rule attached only to a virtual dataset** applies through that dataset, not to the tables beneath it, so a
  chart on one is not checked against mnemiq's own SQL, except on a dataset the build made, where no rule applies to
  the builder (see [Row rules](#row-rules)).
- **No writes to the source database.** Asking writes nothing at all. A build writes only Superset's own charts,
  datasets and dashboards, through the extension, as the builder (see [Building](#building)).
- **No `deep` mode, no MCP door and no Open WebUI endpoint.**
