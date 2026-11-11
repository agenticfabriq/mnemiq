# mnemiq with Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) can talk to mnemiq the way it talks to any OpenAI-compatible
model. Each mnemiq profile appears as one model, and every question is answered as the user who asked it: the tables,
rows and columns that user may read, never a shared service account's. This page covers the settings that make that
true on both sides, and what goes wrong without each one.

It was checked with **Open WebUI v0.13.0, Keycloak 26.0 and oauth2-proxy v7.15.6**. The
[example](../examples/open-webui/README.md) runs that whole stack locally in Docker Compose. Its files are the
settings below, written out.

## Two ways to tell mnemiq who is asking

**Option A (recommended): users sign in to Open WebUI through your identity provider,** and Open WebUI forwards each
user's own access token to mnemiq. mnemiq verifies it as it verifies any token, and nothing about a user reaches a
backend you did not choose.

**Option B: Open WebUI keeps its own accounts** and signs each user into a header with a secret it shares with
mnemiq. Use it only where it is acceptable that, with forwarding on, every backend Open WebUI calls receives each
user's id, email, name and role on every call: other model connections, tool servers, web search, document loaders.
Nothing mnemiq does can prevent that.

## mnemiq's settings

| Setting | Option A | Option B |
|---|---|---|
| `MNEMIQ_OIDC_ISSUER` | your provider's issuer | your provider's issuer, if the workbench or MCP signs users in through it |
| `MNEMIQ_OPENWEBUI_AUDIENCE` | the audience only `/openai/v1` accepts, distinct from `MNEMIQ_OIDC_AUDIENCE` | unset |
| `MNEMIQ_OPENWEBUI_JWT_SECRET` | unset | the same value as Open WebUI's `FORWARD_USER_INFO_HEADER_JWT_SECRET`, at least 32 random bytes |
| `MNEMIQ_OPENWEBUI_CONNECTION_KEY_HEADER`, `MNEMIQ_OPENWEBUI_CONNECTION_KEY` | recommended | required |
| `MNEMIQ_IDENTITY_MAP_PATH` | only for tokens without a roles claim | maps each Open WebUI user id (`sub`) to roles, never an email |
| `MNEMIQ_TENANT` | the tenant every token must carry | the tenant Open WebUI's users belong to |

- **`/openai/v1` is served only where it can verify a caller of its own,** that is, with one of the two settings
  above that name Open WebUI. Otherwise every path under it answers 404, and a startup line says why.
- **A token needs the `db.read` scope** on `/openai/v1`, as on `/v1`, so Open WebUI's client at your provider must
  request it. The signed header carries no scopes: an Open WebUI user's reach is their roles in the identity map.
- **The connection key** is a custom header on mnemiq's connection in Open WebUI. It tells mnemiq a request came
  through that connection, since in option B the same signed token goes to every backend Open WebUI calls. It
  protects against other backends, not against Open WebUI's own admins, who can see it.
- **mnemiq reachable only from Open WebUI and the proxy** is defence in depth behind the connection key.
- **MCP over HTTP stays off in the example** (`MNEMIQ_MCP_TRANSPORT=stdio`): its services talk without TLS, which
  MCP over HTTP refuses. Each MCP client of yours needs its own client at the provider, with an audience mapper adding
  `MNEMIQ_OIDC_AUDIENCE`, the MCP door's URL.

See [settings](settings.md#identity-from-tokens) for every identity setting and what each refuses.

## Your identity provider

The example's [realm](../examples/open-webui/realm.json) is Keycloak's, but the requirements are the provider's:

- **Open WebUI's client (option A):**
  - an audience mapper adding `MNEMIQ_OPENWEBUI_AUDIENCE`, and **only that audience**: no scope shared with the
    workbench's or MCP's clients may add theirs, since a token carrying both is refused;
  - `db.read` among its default scopes;
  - the roles claim mnemiq reads (`realm_access.roles` by default), and a `groups` claim if Open WebUI should place
    users in groups (below);
  - for Microsoft Entra, `offline_access` in its scopes, or the token stops after about an hour;
  - **PKCE (S256) required**, with `OAUTH_CODE_CHALLENGE_METHOD=S256` on Open WebUI;
  - its callback registered **exactly**, `https://<open-webui>/oauth/oidc/login/callback`, and set as Open WebUI's
    `OPENID_REDIRECT_URI` (unset, v0.13.0 sends its legacy `/oauth/oidc/callback`), and `https://<open-webui>/auth`
    registered exactly as its post-logout redirect URI, where its Sign out returns. Not wildcards.

  mnemiq never accepts Open WebUI's client id as an audience: that would accept any token minted for Open WebUI.
- **The workbench's client (the proxy's):** an audience mapper adding `MNEMIQ_OIDC_AUDIENCE`, `db.read`, **PKCE
  (S256) required**, and the workbench's root registered **exactly** as a valid post-logout redirect URI, not a
  wildcard.
- **Token lifetime:** keep the proxy's cookie refresh shorter than it (the example: a 300 s token, a 240 s refresh).

## Open WebUI's settings

- **The connection** to `https://<mnemiq>/openai/v1`:
  - **Option A:** auth `system_oauth`, `ENABLE_OAUTH_ID_TOKEN_COOKIE=false`, forward cookies off, and
    `ENABLE_FORWARD_USER_INFO_HEADERS` **off**: it would add a second credential, which mnemiq refuses.
  - **Option B:** `ENABLE_FORWARD_USER_INFO_HEADERS=true` **and** `FORWARD_USER_INFO_HEADER_JWT_SECRET`. Auth
    `none`, an empty key, and no header that sets `Authorization`, or the static key travels with every request and
    is refused as a second credential. Never forward without the secret: mnemiq refuses the plain headers, so a
    missing secret fails visibly.
  - **Custom headers, both options:** `X-Mnemiq-Task: {{TASK}}` (or your `MNEMIQ_OPENWEBUI_TASK_HEADER`), required,
    since mnemiq refuses a request without it; and the connection key.
  - **The fixed model list (`model_ids`) set to every profile.** Open WebUI keeps one model list per instance, filled
    by whoever fetched it last. Without a fixed list, a profile offered to some roles is "Model not found" for its own
    users whenever another user fetched last. With it, Open WebUI never asks mnemiq for models, so its own access
    grants are the only list, and mnemiq's chat-time 404 refuses a grant made in error.
- **For each mnemiq model:** built-in tools off (Open WebUI otherwise adds them to every browser chat, and the
  endpoint refuses `tools`); no knowledge or files; web search, memory, code interpreter and image generation off;
  an empty system prompt. Grant read access to the groups that mirror the profile's roles. **Never set
  `BYPASS_MODEL_ACCESS_CONTROL`**, which shows every profile to every user.
- **Groups:** with `ENABLE_OAUTH_GROUP_MANAGEMENT=true` and `OAUTH_GROUPS_CLAIM=groups`, Open WebUI places each user
  in the groups their token names, by name, when they sign in. Create the groups first: the example's
  `make-secrets.py users` does.
- **Background generations** (titles, tags, follow-ups, search and retrieval queries) send whole chats, governed rows
  included, to Open WebUI's task model. Either set `TASK_MODEL_EXTERNAL` to a model you trust with those results,
  normally the same endpoint as mnemiq's `MNEMIQ_LLM_*`, or turn them off. The example turns them off. With no task
  model at all, retrieval query generation comes back empty and attached files are silently ignored.
- **Files:** a chat with a file attached under Open WebUI's default retrieval template is refused, with the settings
  to change named in the error.
- **API callers:** `ENABLE_API_KEYS=true`, and the API Keys feature permission granted to their group.
- **Timeouts:** leave `AIOHTTP_CLIENT_TIMEOUT` unset, or above mnemiq's deep-mode budget.

## The workbench behind oauth2-proxy

The [proxy's config](../examples/open-webui/oauth2-proxy.yaml) holds the settings that matter:

- **It injects the access token** (`claimSource: {claim: access_token, prefix: "Bearer "}`), not the ID token,
  whose audience is the proxy's own client. From v7.14 a header value's claim sits under `claimSource:`.
- **`insecureSkipNonce: false`, set explicitly.** Keycloak 26 leaves the nonce out of a refreshed ID token. v7.15.6
  does not check a refreshed session's nonce, so the check can stay on for sign-in. Older releases need it skipped to
  refresh at all, at the cost of the nonce at sign-in.
- **PKCE:** `code_challenge_method: S256`, with the client requiring it.
- **`--whitelist-domain`** names your provider's host as the browser sends it: lowercase, with a port only when it is
  not the scheme's default. It is what lets Sign out go on to the provider's page.
- **The flush interval** lets the workbench's answer stream render as it arrives.
- **`backendLogoutURL`** ends the provider's session on Sign out, while the proxy can load the session.

**A refresh that fails because the provider is unreachable** ends the proxy's session at once if the session was
refreshed before: when a refresh fails the proxy checks the stored ID token's nonce, which Keycloak 26 leaves out of a
refreshed one. On a session's first refresh the stored token is the sign-in's, so the proxy keeps the session and tries
again on each request until that token expires. Either way nobody is let in past the token's lifetime: the browser is
sent to the provider, and once it is back the user signs in there, without a password if the provider's own session
is still alive. A refresh the provider refuses with `invalid_grant` or `invalid_client`, a revoked session for
example, ends the proxy's session at once.

## Sign out

Behind the proxy, the workbench's Sign out ends the proxy's session, and with `MNEMIQ_OIDC_END_SESSION_URL` set it
then sends the browser to your provider's end-session page, which returns it to the workbench's root. The provider's
session ends either way. It asks the user to confirm only when the proxy could not end it: without this setting, that
case would sign the same user straight back in, which on a shared computer hands one user's session to the next.

The URL must be written as a browser sends it, and the three things it needs are in
[settings](settings.md#identity-from-tokens). A user the door refuses with a 403 (no `db.read`, another tenant) is
told the same URL in the refusal's body, so the workbench can let them sign out and switch accounts.

**Open WebUI's own Sign out (option A) ends the provider's session too.** It goes to the provider's end-session
endpoint with the user's ID token, then back to its sign-in page, so the next sign-in asks for the password. Checked
with Open WebUI v0.13.0 and Keycloak 26.0.

## Local development

The example names Keycloak `keycloak.localhost`, which browsers resolve to this machine (checked with Chrome 157 and
Safari 27.2 on macOS), so one address serves the browser and the containers alike, over plain http. A program using
the system's resolver may not resolve it, as on a Linux host with no `myhostname` in its name-service switch, so
`make-secrets.py` reaches Keycloak by its published address.

All of that is for local development only, and so are the proxy's `--cookie-secure=false` and Keycloak's `start-dev`.
A deployment serves its provider over https under its own name, drops `--cookie-secure=false` so the proxy's cookie is
`Secure`, and runs Keycloak with `start`. mnemiq's check allows http only for loopback and `.localhost` names.

## Your own database

The example answers from a small demo database, which its `init` service seeds, enriches and indexes on first start.
To answer from your own database instead:

1. **Describe the source.** Write a source manifest naming the database, the schema and the tables mnemiq may read,
   as in [Sources](../README.md#sources), and mount it into the `init` and `mnemiq` services. From inside a container,
   a database on the machine that runs Docker is `host.docker.internal`; on Linux, also give both services
   `extra_hosts: ["host.docker.internal:host-gateway"]`.
2. **Give mnemiq a user that can only read,** and pass its name and password to both services through an `env_file`,
   as the example passes its other secrets: `MNEMIQ_ORACLE_USER` and `MNEMIQ_ORACLE_PASSWORD` for Oracle, whose
   driver the image includes. [Deploying mnemiq against Oracle](oracle-deployment.md) says what the engine enforces on
   its own and what only the database can.
3. **Grant your tables to the roles.** The example's `authz.json` grants each realm role a few of the demo's tables,
   so until it names yours, no one can read them. List each role's tables as the source stores the names (an Oracle
   table created unquoted is upper case), as [Access policy](access-policy.md) describes.
4. **Point mnemiq at it** in the compose file's `x-mnemiq-env`: `MNEMIQ_SOURCES_PATH` to the manifest, and
   `MNEMIQ_STORE_PATH` to a file under `/app/demo`, the `demo` volume, which the image's own user can write. If you
   have business definitions, set `MNEMIQ_ONTOLOGY_RECORDS_PATH` to them here as well: `mnemiq enrich` reads them.
5. **Enrich and index,** with the model mnemiq will answer with. On a new `demo` volume, the `init` service does it
   on the first start; delete `python scripts/seed_demo.py && ` from its command so that it does not seed the demo
   too. It skips a volume it has already initialized, so after the demo has run, and after any change to the
   definitions or the tables, stop `mnemiq` (only one process should have the store open), run
   `docker compose -f examples/open-webui/compose.yaml run --rm --no-deps mnemiq sh -c "mnemiq enrich && mnemiq build"`,
   and start `mnemiq` again. Enrichment asks the model about every table it describes, so a wide schema takes a while.

At start-up mnemiq says whether the largest prompt your tables can produce fits the model's window. Read that line
before inviting anyone: "fits only by withholding column descriptions" means a question that brings in the widest
tables is sent with fewer descriptions than enrichment wrote, and "TOO SMALL" means such a question can fail.

## What an answer looks like

mnemiq replies in Markdown: the answer and a status line, the SQL in a fenced block, up to 20 rows as a table
("20 of N rows"), and the run id. Only an answer that was checked says "verified". Nothing from data or the model can
render as a link, an image, HTML or math in Open WebUI: those characters arrive backslash-escaped, so a client that
does not render Markdown shows the backslashes. Each finished stage of the work arrives as reasoning, which Open WebUI
shows as a reasoning block while the answer is prepared.

## Licence

From v0.6.6, Open WebUI's licence lets a deployment alter or remove its branding only with 50 or fewer users in a
30-day window, a substantive contributor's written permission, or an enterprise licence. Running it against mnemiq
is unaffected: above 50 users, keep the branding. See [docs.openwebui.com/license](https://docs.openwebui.com/license).
