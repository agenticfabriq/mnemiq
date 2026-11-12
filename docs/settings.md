# Settings

mnemiq is one engine with many ways to run it. This page explains every setting it has, in four words:

- **Technique:** an idea about how to answer a question well, such as letting the SQL writer reason before it writes,
  or showing it the bare schema instead of described tables. Many come from published NL2SQL work and from other
  systems. mnemiq builds each one in its own code.
- **Setting:** one named value. A setting either turns a technique on or tunes it, or is a plain parameter with no
  technique behind it, such as the model's address or how long an answer may take.
- **Profile:** a named set of setting values for one deployment, usually one model on one kind of data. It lists only
  what differs from the deployment's own settings, and only settings a profile may change. See
  [profiles.md](profiles.md).
- **Mode:** `instant`, `thinking` or `deep`. Each mode has its own values for a few settings, and a profile can change
  them mode by mode.

Which techniques help depends on the model and the data, so the way to choose is to measure: run each candidate
profile as one arm of an experiment, beside the others, over questions with known answers.
[Finding the right settings](profiles.md#finding-the-right-settings) says how, and `mnemiq sweep` runs such an
experiment for you: see [sweeps.md](sweeps.md).

## Where a setting is set

- **The deployment** sets a setting in the environment as `MNEMIQ_` and its name in capitals: `retrieval_k` is
  `MNEMIQ_RETRIEVAL_K`. Two names differ: `default_mode` is `MNEMIQ_MODE`, and `verify_override` is `MNEMIQ_VERIFY`.
  A true or false setting takes `1` or `0`.
- **A profile** sets any of the [profile settings](#profile-settings) by name in the profiles file, and the
  [per-mode settings](#per-mode-settings) in a section named for the mode.
- **Without a profiles file,** the per-mode settings go in `MNEMIQ_MODE_SETTINGS` as JSON, for example
  `{"instant": {"card_style": "bare"}}`.

A value set for a mode wins over that mode's own value, a value a profile sets wins over the deployment's, and
anything left unset keeps the default in the tables below.

```toml
[profiles.qwen32b-bare-t0]
roles = ["analyst"]
llm_base_url = "http://127.0.0.1:8000/v1"
llm_model = "qwen2.5-coder-32b"
api_key_env = "LOCAL_LLM_KEY"
writer_temperature = 0.0          # Repeatable sampling

[profiles.qwen32b-bare-t0.instant]
card_style = "bare"               # Schema form, in instant mode only
```

## Techniques

Each technique below names the settings that choose it. The tables after this section give every setting's default.

### Schema form

What the SQL writer is shown about each table, set by `card_style`. `cards`, the default, shows each table's columns,
types and keys with the descriptions and code meanings enrichment found. `ddl` shows `CREATE TABLE` statements, for a
model trained on them. `bare` shows `CREATE TABLE` with names, types and keys alone. Retrieval always searches the
cards, whichever form the writer is shown. A frontier model makes use of the descriptions; a smaller model can do
better with less text in front of it. It can be set per mode.

### Schema retrieval

How many schema cards go into the writer's prompt, picked by how close they are to the question: `retrieval_k`. More
raises the chance that the right table is in front of the writer and makes the prompt longer, so a model with a small
context window may need fewer. The default of 24 was measured on the Spider 2.0-lite benchmark.

### Reasoning before the SQL

`writer_reasoning`. With `none`, the writer replies with its JSON answer alone. With `before_sql`, it is asked to think
the question through step by step first, and the engine reads the JSON object its reply ends with. A reply cut off
before its answer counts as no answer, never as a guess. Reasoning costs output tokens and time on every call, and it
cannot be combined with `guided_sql`, which allows no text before the JSON.

### A plain prompt

`writer_rules`. With `full`, the writer's system prompt carries the engine's persona, its hard rules and its guidance
on coded values. With `none`, it keeps only the task, the instruction for declining and the reply format. The engine's
SQL decider enforces the hard rules either way, so leaving the text out changes what the model reads, not what the
engine allows to run.

### Prompt layout

`prompt_layout`. With `question_first`, the question and its context go in the user message. With `question_last`,
the context goes in the system message and the question alone in the user message.

### Attempting rather than declining

`assertive_sql` tells the writer to attempt an answer instead of declining when it is unsure. It trades declines for
attempts, and some of those attempts are wrong; the decider and the verifier still check every statement.

### Guided decoding

`guided_sql` asks the server to hold the writer's reply to the answer's JSON schema with a non-empty `sql`, through
`response_format`. vLLM's structured output enforces the schema, so there the reply cannot be prose or malformed JSON.
Other OpenAI-compatible providers treat it as best effort, so a malformed reply is still possible there, and it is
declined. It leaves no room for reasoning before the answer.

### Repeatable sampling

`writer_temperature` sets the writer's sampling temperature. At 0 the writer takes its most likely words, so the same
question gets the same SQL run after run on a server that honours it. Unset, none is sent and the server's default
applies (0.7 for Qwen2.5). It is refused for a model that takes no temperature, such as GPT-5. `llm_seed` sends a
sampling seed on every call; vLLM honours it, and the hosted endpoint accepts it and ignores it. A mode set to more
than three candidates, as deep is, raises the writer's temperature to 0.7 for a candidate that repeats a strategy
("Several candidates"), so its answers at 0 repeat run after run only when `llm_seed` is set.

### Repair from feedback

Two loops, set per mode. The SQL decider checks every statement before it runs. When it refuses one, for example over
a column that does not exist, the writer is asked again with the reason, up to `plan_attempts` writer calls. Three
problems are a statement that would run and answer wrong: a logic error the linter recognises, a filter value the data
does not contain, and an aggregate over rows a join has multiplied. For those, the `corrector`, when on, makes one
targeted edit, and the decider checks the result again. When the database rejects a statement the decider passed, the
whole step runs again with the database's error, up to `max_attempts` times. Instant mode makes one attempt and has no
corrector.

### Several candidates

Deep mode writes several candidate statements with different strategies (`candidates`), runs them all and groups
them by their result. `min_agreement` is the share of candidates that must return the same result: below it, or when
some candidates fail to run, the question is declined as disagreement rather than answered with a pick. When
`writer_temperature` is set below 0.7, a candidate whose strategy an earlier candidate already used, which only a mode
set to more than three candidates has (deep's fourth and fifth of five), is sent 0.7, so it samples instead of
repeating that candidate's answer, and the agreement gate counts no copies. Every other candidate, every mode with one
candidate, and the fallback to the single path are sent the temperature as set. When no
candidate runs at all, deep mode falls back to the single path with its repairs, and that answer has no agreement
behind it. When the candidates do not already agree, the `judge`, a model call, picks among their distinct results;
without it, the most common result is taken. Both need more than one candidate.

### Checking the result

The verifier's level is set per mode by `verify`. At `sanity`, a deterministic check declines an empty or null result
unless every value it filters on is found in the data (`verify_sanity` turns that check off). A date needs no lookup
where the engine reads it one way. On DuckDB and Postgres that is an ISO date compared with a `DATE` column, or
starting a range or ending it exclusively on a timestamp column. On Oracle it is a `TO_DATE` with the format
`'YYYY-MM-DD'` or a `DATE` literal, starting a range or ending it exclusively on a `DATE` or `TIMESTAMP` column, since
an Oracle `DATE` holds a time of day; a bare date string there follows the session's date format, so it still declines. At `full`, the sanity
check runs first and then a model judges whether the result answers the question, declining below `verify_threshold`
confidence. In the technical report's replay of 437 BIRD mini-dev answers, the judge at 0.5 cut the wrong-answer rate
from 39.0% to 19.7% and cost 4.5 points of exact match. `verify_grounding` adds a check that every quoted phrase and
four-digit year in the question appears in the SQL. The judge runs on the chat model unless `verify_base_url` and
`verify_model` name another.

### Refusing risky SQL shapes

Checks in the SQL decider that refuse a statement before it runs. `guard_fanout` refuses an aggregate over rows a join
has multiplied, so a total cannot be counted twice, as a refusal the writer can repair. Unset, it is on when the
store's partitions were profiled. `guard_undefined_terms` refuses an answer that rests on a business term the writer
declared and no certified definition covers. It is off and not recommended: the writer declared ordinary words as
terms, and too many good questions were refused.

### The answer's form

The rows go out as soon as they are verified, before any sentence. `prose`, per mode, has a model call turn the rows
into sentences afterwards; off, the rows are the answer and no call is made. `answer_markdown` lets that sentence use
markdown lists and tables when the result has structure.

## Profile settings

The settings a profile may change: how a question is answered, never who may see what.

| Setting | Default | Technique | What it does |
|---|---|---|---|
| `llm_base_url` | unset | plain parameter | The chat model's OpenAI-compatible endpoint. A profile that moves it must name its own key variable. |
| `llm_model` | `"openai.gpt-5.5"` | plain parameter | The chat model's id. |
| `llm_seed` | unset | Repeatable sampling | A sampling seed sent on every call. Deep mode adds each candidate's index. |
| `llm_context_window` | unset | plain parameter | The chat server's context window in tokens, for a server that does not report it (Ollama). vLLM reports its own. |
| `llm_prompt_cut_check` | `true` | plain parameter | Refuses an answer when the server cut the prompt to fit its window without an error. Off only for a server whose token counts leave out cached tokens. |
| `llm_window_check` | `true` | plain parameter | At boot, warns when the longest prompt this store can produce, plus the reply, would not fit the server's window. A warning, never a refusal. |
| `card_style` | `"cards"` | Schema form | `cards`, `ddl` or `bare`: what the writer is shown about each table. |
| `retrieval_k` | `24` | Schema retrieval | How many schema cards go into the writer's prompt. |
| `writer_reasoning` | `"none"` | Reasoning before the SQL | `none`, or `before_sql` to let the writer reason before its JSON answer. |
| `writer_rules` | `"full"` | A plain prompt | `full` carries the engine's rule text in the writer's prompt; `none` leaves it out. |
| `prompt_layout` | `"question_first"` | Prompt layout | `question_first` or `question_last`: where the question sits in the writer's prompt. |
| `assertive_sql` | `false` | Attempting rather than declining | Tells the writer to attempt an answer instead of declining. |
| `guided_sql` | `false` | Guided decoding | Asks for the reply in the JSON schema with a non-empty `sql`: enforced by vLLM, best effort elsewhere. |
| `writer_temperature` | unset | Repeatable sampling | The writer's sampling temperature; unset sends none. |
| `guard_fanout` | unset | Refusing risky SQL shapes | Refuses an aggregate over rows a join multiplied. Unset: on when partitions were profiled. |
| `guard_undefined_terms` | `false` | Refusing risky SQL shapes | Refuses a business term with no certified definition. Not recommended. |
| `verify_sanity` | `true` | Checking the result | The deterministic check on empty and null results, in every mode that verifies. |
| `verify_grounding` | `false` | Checking the result | Declines when a quoted phrase or year in the question is missing from the SQL. |
| `verify_threshold` | `0.5` | Checking the result | The judge's confidence below which an answer is declined. |
| `verify_base_url` | unset | Checking the result | The judge's endpoint; unset uses the chat endpoint. |
| `verify_model` | unset | Checking the result | The judge's model; unset uses the chat model. |
| `answer_markdown` | `false` | The answer's form | Lets the sentence use markdown when the result has structure. |

## Per-mode settings

Each mode's own values. A profile changes them in its mode section; without a profiles file,
`MNEMIQ_MODE_SETTINGS` does.

| Setting | instant | thinking | deep | Technique | What it does |
|---|---|---|---|---|---|
| `card_style` | unset | unset | unset | Schema form | The schema form for this mode; unset uses the profile's `card_style`. |
| `candidates` | `1` | `1` | `5` | Several candidates | How many candidate statements to write and run. |
| `corrector` | `false` | `true` | `true` | Repair from feedback | One targeted edit for a statement the decider refuses as wrong but runnable. |
| `judge` | `false` | `false` | `true` | Several candidates | A model call picks among the candidates' results when they do not agree. |
| `min_agreement` | unset | unset | `0.6` | Several candidates | The share of candidates that must agree, or the question is declined; unset: no gate. When no candidate runs, the single path answers instead. |
| `budget_s` | `30.0` | `120.0` | `300.0` | plain parameter | A soft time budget in seconds. Nothing is cut off when it passes; the sentence is then written as an out-of-budget answer from the rows in hand. |
| `max_attempts` | `1` | `3` | `3` | Repair from feedback | Tries after the database rejects a statement. |
| `verify` | `"sanity"` | `"full"` | `"full"` | Checking the result | The verifier's level: `off`, `sanity` or `full`. |
| `plan_attempts` | `3` | `3` | `3` | Repair from feedback | Writer calls spent on statements the decider refuses (each deep candidate gets one). |
| `prose` | `true` | `true` | `true` | The answer's form | A model call turns the rows into sentences; off, the rows are the answer. |

## Deployment settings

Settings no profile may change: who may see what, where the data lives, and how the deployment runs. They are set in
the environment.

### Access, identity and the network

| Setting | Default | What it does |
|---|---|---|
| `local_only` | `false` | Refuses to start if a chat, embedding, judge or Verity endpoint would leave this machine or network, or a DuckDB extension it loads is not already on disk. |
| `authz_path` | unset | The access policy's JSON file: which roles may see which tables, rows and columns. See [access-policy.md](access-policy.md). |
| `write_enabled` | `false` | Attaches the source read-write, for governed writes. |
| `principal` | unset | The identity's principal id, for the MCP server run on its own. |
| `roles` | unset | The identity's roles, comma-separated, for the MCP server run on its own. |
| `tenant` | unset | The identity's tenant id, for the MCP server run on its own. With a token verifier configured it is required, and every verified token must carry this tenant; `*` accepts any. |

### Identity from tokens

With `oidc_issuer` set, the HTTP API, and MCP over HTTP at `/mcp`, take each request's identity from its own
verified token, and the `principal` and `roles` above are never used. A tenant is always named: `tenant` is required
once any verifier is configured, and every token must carry it, or `*` accepts any. Roles come from the token's roles
claim when it has one, and from the identity map when it has none. A token's groups grant nothing unless the map
names them or `oidc_groups_claim` opts in. Every setting here that a door cannot enforce stops the server at startup,
with a message naming the setting to change. A token needs the `db.read` scope on every `/v1` route and on
`/openai/v1`, and is refused with a 403 without it, so the client that signs users in to the workbench, and Open
WebUI's own client when it forwards each user's token, must request that scope. Open WebUI's signed user header
carries no scopes and is not asked for one: what its users may read comes from their roles in the identity map
alone. An Open WebUI user with no rule in the map is not refused: they are offered only the profiles open to every
role (with no profiles file, the default alone) and asked with no roles, so they read only what the grants give a
caller with no roles, and each question still reaches the engine and its model. On MCP, `db.read` gates `db_read` and `get_schema`, and `db.write` gates `db_write`. A scope only
narrows: what a token may read is still its roles' grants.

**Sign out through the identity provider.** Behind oauth2-proxy, the workbench's Sign out goes to the proxy's own
sign-out, which ends the proxy's session and asks the identity provider to end the user's. When the proxy cannot load
the session at that moment, the provider's session survives, and the next visit signs the same user straight back in.
With `oidc_end_session_url` set, `/v1/whoami` tells a verified caller where the provider's end-session page is, and
Sign out sends the browser on to it, which returns it to the workbench's root. A caller the door verifies but refuses
on `/v1` gets it too, so they can sign out and switch accounts. A 403 for another tenant or a token without `db.read`
has the body `{"detail": "<reason>", "end_session_url": "<the URL>"}`, with `null` when the setting is unset. A 401,
for a caller nobody has verified, is unchanged and names it nowhere. The provider's session then ends either
way; the provider asks the user to confirm only when the proxy could not end it. Three things must be in place.
Without the first, Sign out quietly stops short of the provider. Without either of the others, the user is left on the
provider's error page:

- **The provider's host in oauth2-proxy's `--whitelist-domain`, as the browser sends it**: lowercase, with a port
  only when the URL names one other than its scheme's default, since a browser drops `:443` from an https URL.
  Without it the proxy drops the redirect, logging `Rejecting invalid redirect … domain / port not in whitelist`, and
  sends the browser to its own default page instead, so Sign out is the proxy's alone.
- **The workbench's root, exactly (for example `https://mnemiq.example/`), among the client's valid post-logout
  redirect URIs**, not a wildcard. Without it, Keycloak shows `Invalid redirect uri` instead of returning to the
  workbench.
- **The proxy client's `client_id` in the URL.** Keycloak needs it, or an ID token, which the workbench never holds.
  Without it, Keycloak shows `Missing parameters: id_token_hint`.

In the last two cases, the provider's session has still ended if the proxy has a `backendLogoutURL`, as the
[example](../examples/open-webui/oauth2-proxy.yaml) does, so the next visit asks for a password: the error page is a
dead end, not a session left open.

**Write the URL as the browser sends it.** A browser lowercases a host, drops a default port, removes a leading zero
from a port and rewrites an IP address into its own form, so the check accepts only a URL it would leave unchanged.
What a refusal costs, by case:

- **An uppercase host** is the same host to the browser: lowercase this setting, and the whitelist entry if it was
  written the same way.
- **A default port (`:443`, `:80`) or a port with a leading zero:** drop or rewrite it in this setting, and in the
  whitelist entry if it was written the same way.
- **An IP address other than `127.0.0.1` or `[::1]`, or an internationalised host:** give the provider a hostname,
  and move the issuer, the cookie host, the whitelist entry and this setting to it together.

**The OpenAI-compatible endpoint.** [Open WebUI](open-webui.md) sets it up end to end. `GET /openai/v1/models` and `POST /openai/v1/chat/completions` serve Open WebUI,
or any OpenAI client, each request as its own verified caller, with one model per profile that caller may pick. They
are served only where the endpoint can verify a caller of its own: `openwebui_jwt_secret` (Open WebUI's signed user
header) or `openwebui_audience` (a provider token for this endpoint). Otherwise every path under `/openai/v1` answers
404, and a startup line says why. Every request must carry the header `openwebui_task_header` names: Open WebUI fills
it from `{{TASK}}`, empty for a question, and a request whose value is not empty is a background task, such as a
title, answered empty without asking the engine. The question is the last user message's text, and
`reasoning_effort` picks the mode: `low` instant, `medium` thinking, `high` deep.

| Setting | Default | What it does |
|---|---|---|
| `oidc_issuer` | unset | The identity provider whose tokens the HTTP API and MCP over HTTP verify, matched exactly against the token's `iss`. Unset, the settings identity above serves every caller. |
| `oidc_audience` | unset | The `aud` a token must carry on `/v1` and on MCP. Under `mcp_transport = "http"` it is the MCP door's own URL, `https://<host>/mcp`. |
| `oidc_jwks_url` | unset | Where the provider's signing keys are fetched. Unset, it is `<issuer>/protocol/openid-connect/certs`. |
| `oidc_jwks_file` | unset | A JWKS file read instead of fetching the keys, for a deployment that may not reach the provider. |
| `oidc_tenant_claim` | `"tenant"` | The token claim holding the caller's tenant. |
| `oidc_roles_claim` | `"realm_access.roles"` | The dotted path of the token's roles claim. A token without it takes its roles from the identity map. |
| `oidc_groups_claim` | unset | Opt-in: passes this claim's groups through as the identity's groups, for a provider whose groups only administrators control. Unset, a token's groups grant nothing except through the identity map. |
| `oidc_leeway_s` | `30.0` | Clock leeway, in seconds, on a token's `exp`, `nbf` and `iat`. |
| `oidc_end_session_url` | unset | The identity provider's end-session URL, as the browser reaches it, with the sign-in proxy client's `client_id` where the provider needs it. Set, the workbench's Sign out ends there, through the proxy's `rd`, so the provider's session ends even when the proxy cannot end it. Unset, Sign out is the proxy's alone. https, or http only for `127.0.0.1`, `[::1]`, `localhost` or a name under `.localhost`; its host a lowercase hostname whose last label starts with a letter (no `xn--` label), or `127.0.0.1` or `[::1]`; a port only when it is not the scheme's default, written without a leading zero; printable ASCII with no backslash; it needs `oidc_issuer`, and it must not carry `post_logout_redirect_uri`, which the workbench adds. |
| `openwebui_audience` | unset | The `aud` an identity provider's token must carry on Open WebUI's endpoint, distinct from `oidc_audience`. The token needs the `db.read` scope there too. Unset, that endpoint accepts no Bearer token. |
| `openwebui_jwt_secret` | unset | Turns on Open WebUI's signed user header: the HS256 secret it shares with Open WebUI, at least 32 bytes. |
| `openwebui_jwt_header` | `"X-OpenWebUI-User-Jwt"` | The header carrying Open WebUI's signed user token. |
| `openwebui_jwt_max_lifetime_s` | `300` | The longest lifetime, `exp` minus `iat`, accepted on Open WebUI's signed user token. |
| `openwebui_user_headers` | `"X-OpenWebUI-User-Id,X-OpenWebUI-User-Email,X-OpenWebUI-User-Name,X-OpenWebUI-User-Role"` | Open WebUI's unsigned user headers. A request carrying any of them is refused once an issuer is configured. |
| `openwebui_connection_key_header` | unset | The custom header on mnemiq's Open WebUI connection that carries the connection key. |
| `openwebui_connection_key` | unset | The connection key, at least 32 bytes, which tells mnemiq a request came through its own Open WebUI connection. Required with the signed user header. |
| `openwebui_task_header` | `"X-Mnemiq-Task"` | The custom header on mnemiq's Open WebUI connection that carries `{{TASK}}`. Every `/openai/v1` request must carry it, empty for a chat turn; a request whose value is not empty is a background task, such as title generation, and gets an empty completion without reaching the engine. It must be a header of its own. |
| `identity_map_path` | unset | A JSON file mapping each issuer's subjects, groups and verified emails to roles, for tokens that carry no roles claim. |
| `mcp_transport` | `"stdio"` | `stdio` runs the MCP server for one local caller. `http` serves it at `/mcp` on the HTTP API, each request judged by its own token. |

### The data source

| Setting | Default | What it does |
|---|---|---|
| `pg_dsn` | unset | A Postgres source's connection string. |
| `oracle_user` | unset | An Oracle source's user. |
| `oracle_password` | unset | An Oracle source's password. |
| `oracle_config_dir` | unset | The directory holding `tnsnames.ora` and, for a TLS target, the wallet. See [oracle-deployment.md](oracle-deployment.md). |
| `oracle_wallet_password` | unset | The password for the wallet's `ewallet.pem`. |
| `oracle_pool_max` | `4` | The most pooled Oracle connections; more requests than this wait for one. |
| `oracle_acquire_timeout_s` | `10.0` | Seconds a request waits for a pooled Oracle connection before failing. |
| `oracle_probe_timeout_s` | `30.0` | Seconds allowed for the statements the engine runs about itself (checks, not data queries); 0 turns the limit off. |
| `oracle_read_only_ttl_s` | `300.0` | Seconds a read-only verdict stands before the engine checks the database's open mode again; 0 never checks again. |
| `sources_path` | unset | A JSON manifest of several sources, for a federated store. |
| `source_id` | `"acme"` | The source's id when there is one. |

### The store and what built it

| Setting | Default | What it does |
|---|---|---|
| `store_path` | `"mnemiq.duckdb"` | The DuckDB file holding the semantic store. |
| `control_dsn` | unset | A Postgres database for the shared cache and the store's version pointer. |
| `embed_model` | `"openai.text-embedding-3-small"` | The embedding model. It stays with the deployment, since the store's index was built with it. |
| `embed_base_url` | unset | The embedding endpoint; unset uses the chat endpoint. |
| `embed_api_key` | unset | The embedding endpoint's key; unset uses the chat key. |
| `enrich_facts` | `false` | An experimental enrichment phase that adds table facts to the store. |
| `enrich_examples` | `false` | An experimental enrichment phase that adds verified examples to the store. |
| `dictionary_path` | unset | A data dictionary's JSON file, which grounds what codes mean. See [grounding.md](grounding.md). |
| `ontology_records_path` | unset | Ontology records' JSON file, which binds code schemes and grounds bare codes. |
| `definition_index_max_concepts` | `500` | The most concepts per scheme in the index enrichment builds for grounding definitions. |
| `binding_suggestions_path` | unset | Where enrichment writes its suggested ontology bindings; unset puts them beside the store. |

### Keys

Keys are never written in a profiles file; a profile names the variable that holds its key instead.

| Setting | Default | What it does |
|---|---|---|
| `llm_api_key` | unset | The chat endpoint's key. |
| `verify_api_key` | unset | The judge's key; unset uses the chat key. |

### The verifier

| Setting | Default | What it does |
|---|---|---|
| `verify_override` | unset | `0` turns the verifier off in every mode, `1` sets every mode to `full`; unset keeps each mode's own level. |
| `verify_judge` | `false` | Runs the model judge on the evaluation path. |
| `verify_fail_closed` | `true` | When the judge cannot be reached, the answer stops as a failure instead of going out unchecked. |

### Verity

| Setting | Default | What it does |
|---|---|---|
| `verity_records_url` | unset | The endpoint for certified records, ending `/api/semantic/records/open`. |
| `verity_watermark_path` | unset | Where the incremental pull keeps its watermark; unset puts it beside the store. |
| `verity_page_size` | `500` | Records asked for per page; 0 pulls everything in one request. |
| `verity_full_resync_after_secs` | `86400` | Seconds before the certified records are pulled again in full, which is the only way a withdrawn record leaves. |
| `verity_traces_url` | unset | Where answer traces are sent; unset sends none. |
| `verity_trace_send_text` | `false` | Traces also carry the question, the SQL, the answer and any decline's message. |
| `verity_trace_send_identity_detail` | `false` | Traces also carry the caller's email, roles, groups and attributes. |
| `verity_trace_send_rows` | `false` | Traces also carry the result rows: the data itself. |
| `verity_token_url` | unset | The OAuth2 token endpoint for the client-credentials grant. |
| `verity_client_id` | unset | The service client's id. |
| `verity_client_secret` | unset | The service client's secret, shown only to the token endpoint. |

### Running the deployment

| Setting | Default | What it does |
|---|---|---|
| `profiles_path` | unset | The profiles file. Unset, the deployment's settings run as one profile named `default`. |
| `mode_settings` | unset | Per-mode settings as JSON, when there is no profiles file. |
| `default_mode` | unset | The mode for a request that names none. Unset, it is `thinking`. |
| `max_question_chars` | `4000` | The longest question answered, in characters. A longer one is declined before anything is sent to a model, never cut. |
| `ack_advisories` | unset | Boot advisories to log quietly, comma-separated as `advisory:verdict`. A changed verdict still warns. |
| `acme_data_dir` | unset | The ACME test dataset's directory, for tests. |
| `answer_log_path` | unset | Where each answer's record goes, one JSON line each: profile, version, model and engine commit, with no question, SQL or rows. Unset puts it beside the store; `off` writes none. |
| `answer_log_text` | `false` | The answer log also holds the question and the SQL. Rows are never logged. |
| `engine_commit` | unset | The engine's commit, for a build with no git checkout to read it from. Stamped on every answer. |
