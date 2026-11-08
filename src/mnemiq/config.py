from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import unquote_plus, urlsplit

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, field_validator, Field, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# The Verity endpoint the certified pull needs, as a VALUE. It is the one that accepts `since`
# and returns a `watermark`; its sibling `/api/semantic/records` does neither, so a deployment
# pointed there either 400s on every page (at this pull's default page size) or -- at any limit
# that endpoint tolerates, 0 included, since none is then sent -- drains happily and merges a
# full dump as though it were a delta, forever (M82).
#
# A constant rather than a sentence, and the field's description is BUILT from it. Three rounds
# of review found a phrasing that defeated a prose guard: `/open` appearing somewhere is
# satisfied by a description warning against it, and `/open` appearing FIRST is satisfied by
# "/open is deprecated, set the other one". A rule about how prose goes wrong is prose.
#
# It lives here rather than beside the pull because the description needs it at class-definition
# time, and this module imports nothing from `mnemiq` -- so the pull imports it and there is no
# cycle, where the other direction would drag the contract models into settings import.
CERTIFIED_RECORDS_PATH = "/api/semantic/records/open"

DEFAULT_MODEL = "openai.gpt-5.5"
DEFAULT_EMBED_MODEL = "openai.text-embedding-3-small"
# Measured on the full Spider 2.0-lite sweep against a variance control; see
# tests/test_config.py::test_the_retrieval_k_default_is_the_measured_one.
DEFAULT_RETRIEVAL_K = 24
# The longest question every stage reads whole (M147). One constant, because the Agent falls back to it when a
# construction site passes no limit: a second copy of the number is a second thing to forget.
DEFAULT_MAX_QUESTION_CHARS = 4_000

_SECRET_HINTS = ("api_key", "dsn", "password", "secret")  # fields whose value is never printed in .env.example
# A URL query parameter named like one of these carries a credential: the secret fields' hints, and the words a query
# uses besides (`token`, `auth`, `sig` for a signed URL's signature, and `pwd` and `passwd` for a password).
_CREDENTIAL_HINTS = (*_SECRET_HINTS, "key", "token", "auth", "sig", "pwd", "passwd")
# Names a credential only as a whole word of the name (`pass`, `db_pass`): inside a longer word (`passfile`, `bypass`)
# it names something else.
_CREDENTIAL_WORDS = frozenset({"pass"})
_NAME_WORDS = re.compile(r"[a-z0-9]+")
_WEB_SCHEMES = ("http", "https")
_KEYWORD_PASSWORD = re.compile(r"""(?i)(?:^|\s)password\s*=\s*(?:'(?:[^'\\]|\\.)*'|\S*)""")


def names_a_credential(name: str) -> bool:
    """Whether a URL query parameter called `name` carries a credential."""
    lowered = name.lower()
    return (any(hint in lowered for hint in _CREDENTIAL_HINTS)
            or not _CREDENTIAL_WORDS.isdisjoint(_NAME_WORDS.findall(lowered)))


_HOST_CHARACTERS = re.compile(r"[\w.-]+")  # letters, digits, `_`, `.` and `-`: a name or an IPv4 address


def _reads_as_host(value: str) -> bool:
    """Whether `value`, read the standard way (`urlsplit`), has a netloc that is a host, with a port that parses if it
    has one. A host is letters, digits, `-`, `.` and `_` (a name or an IPv4 address), or a bracketed IPv6 address, its
    zone included."""
    import ipaddress

    try:
        parts = urlsplit(value)
        _ = parts.port  # raises for a port that is not a number
    except ValueError:
        return False
    host = parts.hostname
    if not host:
        return False
    if not parts.netloc.rpartition("@")[2].startswith("["):
        return _HOST_CHARACTERS.fullmatch(host) is not None
    try:
        ipaddress.IPv6Address(host)  # `hostname` drops the brackets and keeps the zone
    except ValueError:
        return False
    return True


def _split_userinfo(value: str) -> tuple[str, str | None, str]:
    """`value` cut as (its `scheme://` or nothing, its user and password or None, what follows their `@`).

    A URL is read the standard way first. When its netloc reads as a host (`_reads_as_host`: host characters or an IP
    literal, and a port that parses if there is one), that reading stands: the user and password are whatever precedes
    the netloc's `@`, and an `@` in the path, query or fragment is theirs. Otherwise (a password holding a raw `/`
    leaves a netloc such as `user:abc`, whose port does not parse, or `AbC+dE`, which is no host) they run to the last
    `@` before whitespace, `?` or `#`, the rule `mnemiq.redact` documents for a source's words, so the password is
    taken whole. With no scheme, an Easy Connect string's `user/password@host` is read by that rule too.
    """
    scheme, sep, rest = value.partition("://")
    if sep and _reads_as_host(value):
        end = re.search(r"[/?#]", rest)
        userinfo, at, _ = (rest[:end.start()] if end else rest).rpartition("@")
        return (scheme + sep, userinfo, rest[len(userinfo) + 1:]) if at else (scheme + sep, None, rest)
    prefix, rest = (scheme + sep, rest) if sep else ("", value)
    end = re.search(r"[\s?#]", rest)
    at = rest.rfind("@", 0, end.start() if end else len(rest))
    return (prefix, rest[:at], rest[at + 1:]) if at != -1 else (prefix, None, rest)


def userinfo_unplaced(value: str) -> bool:
    """Whether `value` is a URL holding an `@` that neither reading places: its netloc does not read as a host
    (`_reads_as_host`), and every `@` comes after whitespace, a `?` or a `#` (a password holding a raw `?` or `#`, or a
    broken URL)."""
    return "://" in value and "@" in value and not _reads_as_host(value) and _split_userinfo(value)[1] is None


def without_credentials(value: str) -> str:
    """A URL or DSN as a message, a log or a fingerprint may show it.

    No user or password before the host's `@` (a database URL keeps its user, which names who reads and is no secret),
    as `_split_userinfo` places them; no query parameter named like a credential; and no `password=` in a keyword DSN.
    Everything else stays as written. So a URL whose netloc reads as a host and port comes back unchanged unless it
    holds a credential, an `@` in its path, query or fragment included. A URL whose netloc does not read loses
    everything up to the last `@` before whitespace, `?` or `#`, its credentials with it, even when no parser accepts
    it.
    """
    if "://" not in value and _KEYWORD_PASSWORD.search(value):
        return " ".join(_KEYWORD_PASSWORD.sub(" ", value).split())
    if "://" not in value and any(c.isspace() for c in value):
        return value  # a keyword DSN with no password in it
    prefix, userinfo, rest = _split_userinfo(value)
    kept_user = ""
    if userinfo is not None:
        # A database URL keeps its user, which names who reads; a web URL's user is part of its credential. A "user"
        # holding a `/` is no user: the read ran across a path.
        user = userinfo.split(":", 1)[0]
        if prefix and prefix[:-3].lower() not in _WEB_SCHEMES and user and "/" not in user:
            kept_user = f"{user}@"
    before_fragment, hash_, fragment = rest.partition("#")
    path, question, query = before_fragment.partition("?")
    if question:
        pairs = query.split("&")
        kept = [pair for pair in pairs if not names_a_credential(unquote_plus(pair.split("=", 1)[0]))]
        if len(kept) < len(pairs):
            before_fragment = path + ("?" + "&".join(kept) if kept else "")
    return f"{prefix}{kept_user}{before_fragment}{hash_}{fragment}"


_HOST_IN_USERINFO = ("the only host it gives is read from inside its user and password, which are not percent-encoded, "
                     "so it is not named")


def _without_userinfo(text: str, url: str) -> str:
    """`text`, a parser's message about `url`, with whatever `url` holds before its host's `@` taken out: a parse error
    can quote the URL's netloc."""
    userinfo = _split_userinfo(url)[1]
    return text if userinfo is None else text.replace(f"{userinfo}@", "")


# Every `Settings` field `assert_local_only` checks, by name. Nothing enforces that a NEW field
# whose name ends in `_url` gets added here -- a future verity_*_url (or any other) setting would
# otherwise be silently exempt while the run still prints "verified". That is what
# tests/test_local_only.py::test_every_url_field_is_checked_or_explicitly_exempted is for: it
# fails on any `_url` field that is neither in this tuple nor in that test's own EXEMPTED set,
# which is where a field would go instead of here if checking it were ever wrong (there are none
# today; MNEMIQ_LOCAL_ONLY's whole premise is that every network-facing URL is accounted for).
_LOCAL_ONLY_CHECKED_URL_FIELDS = (
    "llm_base_url", "embed_base_url", "verify_base_url",
    "verity_records_url", "verity_token_url", "verity_traces_url",
)


@dataclass(frozen=True)
class SourceSpec:
    id: str
    kind: str  # one of adapters.resolve.KINDS: "postgres" | "sqlite" | "duckdb" | "oracle"
    target: str  # dsn (postgres), file path (sqlite/duckdb), Easy Connect or TNS alias (oracle)
    catalog: str  # DuckDB attach alias AND the object_id prefix in a federated store
    schema: str  # source schema: "public" (postgres) / "main" (sqlite)
    # The tables enrichment may see, exact names or shell patterns, without regard to case (M110);
    # empty means every table the source reports. See `mnemiq.adapters.scoped`.
    tables: tuple[str, ...] = ()


def _manifest_tables(entry: dict) -> tuple[str, ...]:
    """A source's optional `tables` list, refused unless it is non-empty strings: a bare string
    would iterate as one-letter patterns, and an empty list would read as 'no tables'."""
    tables = entry.get("tables")
    if tables is None:
        return ()
    if (not isinstance(tables, list) or not tables
            or not all(isinstance(t, str) and t.strip() for t in tables)):
        raise ValueError(
            f"source {entry.get('id')!r}: \"tables\" must be a non-empty list of table names or "
            f"patterns, got {tables!r}")
    return tuple(t.strip() for t in tables)


class ModeSettings(BaseModel):
    """One mode's settings, as an operator sets them: `agent.modes.modes_for` applies what is set here over the
    mode's own value. Refused rather than ignored, each of them: a benchmark arm whose setting never ran scores
    cleanly and reads as "no effect", a finding about nothing."""

    model_config = ConfigDict(extra="forbid")

    card_style: Literal["cards", "ddl", "bare"] | None = None
    candidates: int | None = Field(default=None, ge=1)
    corrector: bool | None = None
    judge: bool | None = None
    min_agreement: float | None = Field(default=None, ge=0, le=1)
    budget_s: float | None = Field(default=None, gt=0)
    max_attempts: int | None = Field(default=None, ge=1)
    verify: Literal["off", "sanity", "full"] | None = None
    plan_attempts: int | None = Field(default=None, ge=1)
    prose: bool | None = None

    @model_validator(mode="after")
    def _null_only_where_it_means_something(self):
        # `card_style` null is "this deployment's form", `min_agreement` null is "no agreement gate"; on any other
        # setting a null names no value, so it could only be a mistake for leaving the setting out.
        for name in self.model_fields_set - {"card_style", "min_agreement"}:
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null: leave it out to keep the mode's own value")
        return self


class Settings(BaseSettings):
    """All engine configuration, one source of truth. `env_prefix="MNEMIQ_"` maps each field to its
    env var (llm_base_url -> MNEMIQ_LLM_BASE_URL), so every historical name is preserved. No env_file:
    Settings() reads os.environ only (scripts `source .env`), which also keeps tests isolated."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMIQ_", extra="ignore", populate_by_name=True)

    # --- safety ---
    # Scoped to HTTP endpoints deliberately, not every network setting: llm/embed/verify_base_url
    # and the three verity_* URLs are all ordinary URLs a hostname can be read from. pg_dsn and
    # control_dsn are NOT checked -- libpq's keyword form, multi-host URIs and unix sockets are
    # not URLs `urlparse` can read a hostname from, and the raised message says so explicitly
    # rather than let its silence imply completeness. A libpq-aware DSN check is separate work.
    local_only: bool = Field(
        default=False,
        description="refuse to start if a chat, embedding, judge or Verity endpoint would leave "
                     "this machine or network, or a DuckDB extension it loads is not pre-seeded "
                     "(`mnemiq extensions fetch`; see assert_local_only for exactly what this "
                     "checks and what it deliberately does not)",
    )
    # --- connection ---
    llm_base_url: str | None = Field(default=None, description="chat/generation OpenAI-compat base URL")
    llm_api_key: str | None = Field(default=None, description="chat/generation API key")
    llm_model: str | None = Field(default=None, description="chat/generation model id")
    pg_dsn: str | None = Field(default=None, description="source Postgres DSN")
    oracle_user: str | None = Field(default=None, description="source Oracle user (Easy Connect carries no credentials)")
    oracle_password: str | None = Field(default=None, description="source Oracle password")
    oracle_config_dir: str | None = Field(default=None, description="directory holding tnsnames.ora and, for a TLS target, the wallet; on-prem TNS aliases and Autonomous mTLS both use it")
    oracle_wallet_password: str | None = Field(default=None, description="password for the wallet's ewallet.pem (thin mode cannot use cwallet.sso)")
    oracle_pool_max: int = Field(default=4, description="max pooled Oracle connections; concurrency above this queues for one, bounded by oracle_acquire_timeout_s")
    oracle_acquire_timeout_s: float = Field(default=10.0, description="how long a request waits for a pooled Oracle connection before failing; a bounded wait, because the alternative is every worker blocking on one stalled statement")
    oracle_probe_timeout_s: float = Field(default=30.0, description="call timeout for Oracle statements the engine issues about ITSELF -- validation parses and boot advisories -- which have no legitimate reason to run long; 0 disables. Data queries are NOT bounded by this")
    oracle_read_only_ttl_s: float = Field(default=300.0, description="how long a `constrained` read-only verdict stands before the adapter re-probes the database's open mode; the window is the exposure, since a database reopened READ WRITE reopens M66's PL/SQL write path. 0 disables re-probing")
    ack_advisories: str | None = Field(default=None, description="boot advisories to log at INFO instead of WARNING, comma-separated as <advisory>:<verdict> (e.g. read-only-basis:unverifiable, window:too-small, window:trims); a CHANGED verdict still warns")
    acme_data_dir: str | None = Field(default=None, description="ACME golden dataset dir (tests)")
    embed_model: str | None = Field(default=None, description="embedding model id")
    embed_base_url: str | None = Field(default=None, description="embedding endpoint (defaults to chat)")
    embed_api_key: str | None = Field(default=None, description="embedding API key (defaults to chat)")
    # --- store / sources / identity defaults ---
    authz_path: str | None = Field(default=None, description="local authz grants JSON path")
    sources_path: str | None = Field(default=None, description="multi-source manifest JSON path")
    control_dsn: str | None = Field(default=None, description="control Postgres DSN (L2 cache + version pointer)")
    source_id: str | None = Field(default=None, description="single-source id")
    store_path: str | None = Field(default=None, description="DuckDB semantic store path")
    default_mode: str | None = Field(default=None, validation_alias="MNEMIQ_MODE",
                                     description="default answer mode: instant|thinking|deep")
    write_enabled: bool = Field(default=False, description="attach the source read-write (governed writes)")
    # --- behavior levers / MCP standalone identity ---
    # Validated, not free text: `style != "ddl"` in the renderer means `DDL` or `ddl ` would
    # silently select the OTHER form, and the run record would not say which one produced the
    # score -- a 35.7% form and a 50.5% form told apart only by shell history.
    card_style: Literal["cards", "ddl", "bare"] = Field(default="cards",
                            description="schema card form for the GENERATOR: cards|ddl|bare. "
                                        "`ddl` renders CREATE TABLE for a model fine-tuned "
                                        "on DDL; `bare` is CREATE TABLE with names, types and "
                                        "keys alone; the retrieval index always embeds `cards`.")
    profiles_path: str | None = Field(default=None,
                            description="TOML file of named profiles, each a model and its settings per mode, and the "
                                        "roles that may pick it; unset runs these settings as one default profile")
    # NoDecode, read by `_read_mode_settings`: pydantic-settings would otherwise JSON-decode the raw variable and fail
    # on the empty line a copied `.env.example` holds, before any validator could say what it means.
    mode_settings: Annotated[dict[str, ModeSettings] | None, NoDecode] = Field(default=None,
                            description="per-mode settings as JSON, e.g. {\"instant\": {\"card_style\": \"bare\"}}: "
                                        "card_style, candidates, corrector, judge, min_agreement, budget_s, "
                                        "max_attempts, verify, plan_attempts, prose; unset keeps the mode's own. With MNEMIQ_PROFILES_PATH, state them in the file.")

    @field_validator("mode_settings", mode="before")
    @classmethod
    def _read_mode_settings(cls, value):
        if isinstance(value, str):
            return json.loads(value) if value.strip() else None
        return value

    writer_temperature: float | None = Field(default=None, ge=0, description="the SQL writer's sampling temperature; unset sends none, so the server's default applies (Qwen2.5: 0.7). Refused for a model that takes none, such as GPT-5, which would ignore it")
    prompt_layout: Literal["question_first", "question_last"] = Field(default="question_first", description="how the writer's prompt is arranged: question_first sends the question and its context as the user message; question_last puts the context in the system message and the question alone in the user message")
    writer_reasoning: Literal["none", "before_sql"] = Field(default="none", description="room for the SQL writer to reason: none asks for the JSON reply alone; before_sql asks it to think the question through step by step first, and reads the last JSON object of its reply. Refused with guided_sql, which allows no text before the JSON")
    writer_rules: Literal["full", "none"] = Field(default="full", description="the rule text in the SQL writer's system prompt: full carries the persona, the hard rules and the coded-values guidance; none keeps only the task, the decline instruction and the reply format. The decider enforces the hard rules either way")
    guided_sql: bool = Field(default=False, description="constrained decoding: force non-empty sql (response_format json_schema; works on vLLM and OpenAI-compatible endpoints)")
    assertive_sql: bool = Field(default=False, description="assertive prompt: attempt an answer instead of deferring")
    # M35, off by default: withdrawn on its own pre-registered criterion. Beacon's answerable band
    # deferred 12 of 24 against a prior of 0 in 144, where the threshold named in advance was 2-3%.
    # See `plan_query` for what the number decomposes into and what reviving it would take.
    guard_undefined_terms: bool = Field(default=False, description="M35: refuse a declared business term with no certified definition (withdrawn -- see plan_query)")
    # M109. Unset means auto: on for a snapshot whose partitions were profiled, off for one
    # enriched before that, where current-row SCD joins would be refused until re-enrichment --
    # so an upgrade changes nothing until the snapshot can support the guard. A pre-registered
    # three-run measurement cleared both bars with it on: repairs on a multi-fact schema (net
    # +6, none broken, every run) and no harm on BIRD (mean broken 0.0 against 1.0). An explicit
    # 1 or 0 wins over auto. The default is pinned by a test.
    guard_fanout: bool | None = Field(default=None, description="M109: refuse an aggregate over rows a join has multiplied, as a repairable refusal; unset or auto = on when the snapshot's partitions were profiled, 1 = always, 0 = never")
    answer_markdown: bool = Field(default=False, description="let the answer use markdown (lists, tables) when the result has structure")
    enrich_facts: bool = Field(default=False, description="eval: table-facts enrichment phase (plan-20, default off)")
    enrich_examples: bool = Field(default=False, description="eval: verified-example enrichment phase (plan-20, default off)")
    dictionary_path: str | None = Field(default=None, description="operator code data-dictionary JSON path (grounds code meanings)")
    ontology_records_path: str | None = Field(default=None, description="ontology records JSON path (binds code schemes, grounds bare codes)")
    verity_records_url: str | None = Field(default=None, description=f"Verity GET {CERTIFIED_RECORDS_PATH} endpoint (governed certified records). NOT the sibling without `/open`, and DO NOT change the page size to make that one fit: it accepts no `since`, so ANY page size it tolerates -- 0 included -- makes every pull a full dump merged as though it were a delta, silently, while this pull's default 500 makes it refuse outright. Loud is the better failure. Verity's published OpenAPI is where that endpoint's missing `since` and its own limit ceiling are declared")
    verity_watermark_path: str | None = Field(default=None, description="path to the Verity /open incremental-sync watermark sidecar JSON (defaults beside the store)")
    verity_page_size: int = Field(default=500, ge=0, description="Verity /open page size sent as ?limit= (0 = unbounded full dump: no ?limit is sent at all). NOT a way to make the sibling endpoint work -- it accepts no `since` at any page size, so lowering this to its ceiling or to 0 turns a loud 400 into a pull that has silently stopped being incremental")
    verity_full_resync_after_secs: int = Field(default=86400, ge=0, description="how stale the locally merged certified set may get before it is re-pulled in full; a withdrawal is invisible to an incremental delta, so this window is the only thing that removes one locally (0 = always full)")
    verity_traces_url: str | None = Field(default=None, description="Verity POST /api/traces/batch endpoint; unset = emit nothing (the engine is unchanged)")
    # The disclosure tiers. Default-closed, and enforced in the EMITTER rather than by Verity
    # discarding on receipt: filtering at the receiver means the bytes already crossed the wire.
    verity_trace_send_text: bool = Field(default=False, description="opt-in: send question text, executed SQL, the answer prose and the deferral MESSAGE (all carry literals from the question and values from the database)")
    verity_trace_send_identity_detail: bool = Field(default=False, description="opt-in: send email, roles, groups and attributes; none is needed for audit and `attributes` is an open dict a deployment fills")
    verity_trace_send_rows: bool = Field(default=False, description="deepest opt-in: send result rows -- the customer's data itself, not a description of it")
    verity_token_url: str | None = Field(default=None, description="Keycloak POST .../realms/<realm>/protocol/openid-connect/token endpoint (OAuth2 client-credentials grant)")
    verity_client_id: str | None = Field(default=None, description="Verity service client id (created in the workbench API Credentials page)")
    verity_client_secret: str | None = Field(default=None, description="Verity service client secret; presented only to the token endpoint, never to the records endpoint")
    llm_seed: int | None = Field(default=None, description="sampling seed forwarded to the provider on every call; honoured by vLLM, IGNORED by the hosted endpoint (accepted, no system_fingerprint, output still varies) -- so it buys reproducibility on the local path only. Deep mode adds each candidate's index, so its 4th and 5th candidates, which repeat a strategy's prompt, do not repeat its sample")
    llm_prompt_cut_check: bool = Field(default=True, description="refuse an answer when the server cut the prompt to fit its context window without an error, as Ollama does (M119): a suspicious prompt-token count (far too few for the characters sent, or on a window size) that does not grow when the prompt is re-sent with padding; 0 only for a server or proxy whose prompt_tokens leaves out tokens it reused from a cache")
    llm_window_check: bool = Field(default=True, description="at boot, warn when the largest generation prompt this store can produce, plus the generator's reply budget, is longer than the server's context window (M119), for each role the access policy declares and from only what that role is shown: counted by the server where it can (vLLM's /tokenize), else bounded by its bytes against MNEMIQ_LLM_CONTEXT_WINDOW; a warning only, never a refusal")
    llm_context_window: int | None = Field(default=None, ge=1, description="the chat server's context window in tokens, for servers that do not report it (Ollama: your OLLAMA_CONTEXT_LENGTH or num_ctx); vLLM reports its own --max-model-len and this is then ignored")
    max_question_chars: int = Field(default=DEFAULT_MAX_QUESTION_CHARS, ge=1, description="the longest question, in characters, that mnemiq answers; every stage reads it whole, and a longer one is declined before retrieval, so no server is sent it (reason question_too_long), never cut (M147: a 500-character cut in the generator alone answered long questions as cut ones). The boot window check budgets this many tokens for the question, so lower it where the model's window is small")
    retrieval_k: int = Field(default=DEFAULT_RETRIEVAL_K, ge=1, description="schema cards retrieved into the generation packet")
    definition_index_max_concepts: int = Field(default=500, ge=0,
        description="per-scheme concept cap for the enrich-time definition-grounding index")
    binding_suggestions_path: str | None = Field(default=None,
        description="where enrich writes ontology binding-suggestions (defaults beside the store)")
    principal: str | None = Field(default=None, description="MCP standalone identity: principal id")
    roles: str | None = Field(default=None, description="MCP standalone identity: comma-separated roles")
    tenant: str | None = Field(default=None, description="MCP standalone identity: tenant id")
    # --- verifier ---
    answer_log_path: str | None = Field(default=None, description="where each answer's record goes: which profile, version, model and engine commit produced it, as one JSON line each, with no question text, SQL or rows; unset keeps it beside the store as <store_path>.answers.jsonl, `off` writes none. Not rotated")
    answer_log_text: bool = Field(default=False, description="opt-in: the answer log also holds the question and the SQL as asked, which carry literals from the question and values from the database; rows are never logged")
    engine_commit: str | None = Field(default=None, description="the commit this engine runs, for a build with no git checkout to read it from; stamped on every answer beside the profile's version")
    verify_override: str | None = Field(default=None, validation_alias="MNEMIQ_VERIFY",
                                        description="verify override: unset=mode defaults, 0=force off, 1=force full")
    verify_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="judge-confidence defer cutoff")
    verify_sanity: bool = Field(default=True, description="deterministic empty/null layer, in every mode a verifier runs in, on the product and eval paths alike: declines an empty or null result unless every string value it filters on is grounded in the value index (M146); 0 answers every such result, the judge kept where a mode runs one")
    verify_grounding: bool = Field(default=False, description="question-literal grounding layer (opt-in)")
    verify_judge: bool = Field(default=False, description="LLM judge layer (eval path)")
    verify_fail_closed: bool = Field(default=True, description="an unreachable judge stops the answer (recorded as a failure, not a deferral) instead of returning it unchecked; 0 restores the fail-open behaviour")
    verify_base_url: str | None = Field(default=None, description="judge endpoint (defaults to chat)")
    verify_model: str | None = Field(default=None, description="judge model (defaults to chat)")
    verify_api_key: str | None = Field(default=None, description="judge API key (defaults to chat)")

    @model_validator(mode="after")
    def _fill_defaults(self) -> "Settings":
        if not self.llm_model:
            self.llm_model = DEFAULT_MODEL
        if not self.embed_model:
            self.embed_model = DEFAULT_EMBED_MODEL
        if not self.source_id:
            self.source_id = "acme"
        if not self.store_path:
            self.store_path = "mnemiq.duckdb"
        return self

    # NB: default_mode is validated at boot by build_runtime (raising UnknownMode with the valid
    # set) -- its established contract. We don't duplicate it here.

    @property
    def verify(self) -> bool:  # eval back-compat: MNEMIQ_VERIFY=1 means "verify on"
        return self.verify_override == "1"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()

    def local_only_offenders(self) -> list[str]:
        """The endpoints that would leave this network under MNEMIQ_LOCAL_ONLY; empty when it is off. For a profile's
        resolved copy, whose endpoints `assert_local_only` on the deployment's settings never sees."""
        return self._local_only_walk()[0] if self.local_only else []

    def _local_only_walk(self) -> tuple[list[str], list[str]]:
        """(offenders, cleared), each endpoint judged independently of the others."""
        import ipaddress
        from urllib.parse import urlparse

        rfc1918 = (
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
        )
        # RFC 4193 unique-local addresses -- IPv6's own analog of RFC1918, not merely something
        # `is_private` happens to also cover. fc00::/7 spans both the locally-assigned fd00::/8
        # in practice and the (never yet used) centrally-assigned fc00::/8 half of the range.
        ula = ipaddress.ip_network("fc00::/7")

        offenders: list[str] = []
        checked: list[str] = []
        for name in _LOCAL_ONLY_CHECKED_URL_FIELDS:
            url = getattr(self, name, None)
            if not url:
                continue
            # Named without its credentials: this message reaches stderr and any log the operator keeps.
            shown = without_credentials(url)
            try:
                host = urlparse(url).hostname
            except ValueError as exc:
                # `urlparse` raises for a malformed IPv6 host literal (e.g. an unterminated
                # `[`) BEFORE `.hostname` is reached -- a different failure point than the
                # `host is None` case below, and one an earlier version of this method let
                # propagate uncaught, discarding every offender already collected for a field
                # visited before this one in loop order.
                offenders.append(f"{name}={shown!r} (not a parseable URL: {_without_userinfo(str(exc), url)})")
                continue
            if host is None:
                offenders.append(f"{name}={shown!r} (no host could be parsed)")
                continue
            if host in ("localhost", "localhost.localdomain"):
                checked.append(name)
                continue
            # Where the netloc does not read as a host and port (`_reads_as_host`), a parser can read a password holding
            # a raw `/` as the host and a path, so the host it gives can be part of the credential; a host found inside
            # the user and password is then not named. Where the netloc reads, its host is named. The decision is the
            # same either way.
            userinfo = None if _reads_as_host(url) else _split_userinfo(url)[1]
            if userinfo is not None and host in userinfo.lower():
                host_named = None
            else:
                host_named = host
            try:
                addr = ipaddress.ip_address(host)
            except ValueError:
                offenders.append(
                    f"{name}={shown!r} (host {host_named!r} is a DNS name; this check does not "
                    "resolve names, so it cannot confirm one stays on this network)"
                    if host_named is not None else f"{name}={shown!r} ({_HOST_IN_USERINFO})"
                )
                continue
            # An IPv4-mapped IPv6 literal (`::ffff:10.0.0.1`) is IPv4's 10.0.0.1 by another
            # spelling -- an IPv6-preferring resolver or proxy can hand one back for an ordinary
            # RFC1918 host, and `addr.version` alone would read it as IPv6 and refuse it. Judge
            # RFC1918 membership on the address it actually maps to; `ipv4_mapped` is None (and
            # this is a no-op) for anything that is not one.
            mapped = getattr(addr, "ipv4_mapped", None)
            v4_equivalent = mapped if mapped is not None else addr
            in_rfc1918 = v4_equivalent.version == 4 and any(v4_equivalent in net for net in rfc1918)
            in_ula = addr.version == 6 and addr in ula
            if not (v4_equivalent.is_loopback or in_rfc1918 or in_ula):
                offenders.append(f"{name}={shown!r} (host {host_named} is publicly routable)"
                                 if host_named is not None else f"{name}={shown!r} ({_HOST_IN_USERINFO})")
                continue
            checked.append(name)
        return offenders, checked

    def assert_local_only(self, *, also: tuple[str, ...] = ()) -> None:
        """Refuse to run when a chat, embedding, judge or Verity endpoint would carry data off
        the network.

        For a deployment whose whole premise is that nothing leaves, a mistyped base URL is not
        a misconfiguration that fails -- it is one that SUCCEEDS, quietly, having sent every
        schema card to a third party. This turns that into a startup error.

        Checked: llm_base_url, embed_base_url, verify_base_url, and the three verity_* endpoints
        (verity_records_url, verity_token_url, verity_traces_url) -- trace_sink POSTs to
        verity_traces_url on every ask/serve, and even with the text/rows disclosure opt-ins off
        that body still carries question hashes, timings, policy hashes and identity, so leaving
        it unchecked would be telemetry escaping under a flag named LOCAL_ONLY. NOT checked:
        pg_dsn and control_dsn -- see the trailing line of the raised message for why, which is
        the answer an operator actually needs, not a comment only a reader of this source sees.

        Loopback, the RFC1918 private ranges, and their IPv6 counterparts pass, because a data
        centre runs the model on another host on its own network -- fc00::/7 (RFC 4193's
        "unique local address" range, which `fd00::/8` in practice always comes from) is
        RFC1918's own IPv6 analog, not a narrower or looser thing, so an IPv6-only deployment
        gets the same allowance an IPv4 one does. An IPv4-mapped IPv6 literal (`::ffff:10.0.0.1`)
        is judged on the IPv4 address it maps to, for the same reason: it is that address, just
        spelled by a resolver or proxy that prefers IPv6 syntax. This is deliberately narrower
        than `ipaddress`'s own `is_private`, which also accepts several IANA special-purpose
        ranges we do NOT want to wave through silently -- 169.254.169.254 above all, the
        link-local address several cloud providers use to serve their metadata API, which is
        exactly the kind of off-box hop this check exists to catch. Anything we cannot place
        inside loopback, RFC1918 or its IPv6 analog is refused:
        a URL `urlparse` itself cannot parse (a malformed IPv6 host literal raises INSIDE
        `urlparse`, before `.hostname` is ever read -- not the same failure as `.hostname`
        returning None, and both have to be caught, not just the second one), an unresolved DNS
        name (public or internal -- this performs no lookup, so an internal name is refused for
        the same reason a public one is, not because it looks suspicious), and every other IANA
        special range are all refused the same way, because none of them is evidence of being on
        this network. Every offender is collected and named together, and each is evaluated
        independently of the others -- one endpoint's `urlparse` failure must not discard an
        offender already found for an earlier one, which is exactly the bug an unguarded
        `urlparse(url).hostname` produced. An operator who fixes one and reruns only to be told
        about the next will conclude the check itself is flaky and disable it.

        A passing, enforced check still writes one line to stderr naming what it verified.
        Silence otherwise means two things that must not look alike: "enforced, verified clean"
        and "MNEMIQ_LOCALONLY (no underscore) was typo'd, so `local_only` is False and this
        method returned on its first line." A typo is inert in exactly the same way a correct,
        clean run is, and only one of them is safe.
        """
        if not self.local_only:
            return
        import sys

        offenders, checked = self._local_only_walk()
        if offenders:
            raise RuntimeError(
                "MNEMIQ_LOCAL_ONLY is set and these endpoints would leave this network:\n  "
                + "\n  ".join(offenders)
                + "\n\nFix: point each of these at a loopback address, an RFC1918 private range, "
                  "or an RFC 4193 IPv6 unique-local address (fc00::/7, in practice fd00::/8), "
                  "or unset MNEMIQ_LOCAL_ONLY to disable this check.\n"
                  "Not checked here: pg_dsn and control_dsn. libpq accepts keyword form "
                  "(`host=... port=...`), multi-host URIs and unix-socket targets, none of "
                  "which `urlparse` can read as a hostname -- a fail-closed check on them would "
                  "refuse every legitimate keyword DSN. Verify those separately."
            )
        # **M107.** Every DuckDB extension this deployment loads -- the store's two and each
        # source's scanner -- is loaded once here, so a machine missing one refuses before the
        # verified line rather than after it, mid-build. A manifest that cannot be read is left
        # to fail where it always has, with its own message; the store's two are checked anyway.
        from mnemiq.duckdb_extensions import SOURCE_EXTENSIONS, STORE_EXTENSIONS, check_extensions

        try:
            kinds = {spec.kind for spec in self.source_specs()}
        except (OSError, ValueError, KeyError, TypeError):
            kinds = set()
        # `also`: what the command loads beyond the deployment's sources (Codex's review of M107).
        loaded = check_extensions(
            [*STORE_EXTENSIONS, *(SOURCE_EXTENSIONS[k] for k in sorted(kinds) if k in SOURCE_EXTENSIONS),
             *also])
        cleared = ", ".join(checked) if checked else "(no chat/embedding/judge/verity endpoint configured)"
        print(
            f"MNEMIQ_LOCAL_ONLY verified: {cleared}; DuckDB extensions loaded, none downloaded: "
            f"{', '.join(sorted(loaded))} -- pg_dsn/control_dsn are not checked here",
            file=sys.stderr,
        )

    def embed_endpoint(self) -> tuple[str | None, str | None]:
        """The endpoint embeddings use. Defaults to the chat endpoint; set MNEMIQ_EMBED_BASE_URL/KEY to
        hold embeddings on a separate endpoint while chat/generation points elsewhere."""
        return (self.embed_base_url or self.llm_base_url, self.embed_api_key or self.llm_api_key)

    def verify_endpoint(self) -> tuple[str | None, str | None]:
        """The endpoint the verifier's judge uses. Defaults to the chat endpoint; set
        MNEMIQ_VERIFY_BASE_URL/KEY to run the judge on a separate (e.g. local) model."""
        return (self.verify_base_url or self.llm_base_url, self.verify_api_key or self.llm_api_key)

    def source_specs(self) -> list["SourceSpec"]:
        """Resolved source list. A manifest wins; otherwise synthesize the single legacy source from
        source_id + pg_dsn (today's behavior -> exactly one spec)."""
        if self.sources_path:
            with open(self.sources_path) as fh:
                raw = json.load(fh)
            return [SourceSpec(id=d["id"], kind=d["kind"], target=d["target"],
                               catalog=d["catalog"], schema=d["schema"],
                               tables=_manifest_tables(d)) for d in raw]
        return [SourceSpec(id=self.source_id or "acme", kind="postgres",
                           target=self.pg_dsn or "", catalog="src", schema="public")]

    @field_validator("guard_fanout", mode="before")
    @classmethod
    def _auto_is_unset(cls, value):
        # `.env.example` renders an unset default as `MNEMIQ_GUARD_FANOUT=`, and a sourced copy
        # sets the variable to "" -- which a plain optional bool refuses, failing every boot.
        if isinstance(value, str) and value.strip().lower() in ("", "auto"):
            return None
        return value

    @classmethod
    def env_example(cls) -> str:
        """A `.env.example` template rendered from the schema: one MNEMIQ_<NAME>=<default> line per
        field with its description. Secrets render with an empty value (never a real one)."""
        lines = ["# mnemiq configuration -- generated by `mnemiq config`. Copy to .env.",
                 "# Secrets show empty; fill them in your own .env (never commit real values).", ""]
        for name, info in cls.model_fields.items():
            env = (info.validation_alias if isinstance(info.validation_alias, str)
                   else f"MNEMIQ_{name.upper()}")
            secret = any(h in name for h in _SECRET_HINTS)
            default = "" if (secret or info.default is None) else info.default
            desc = info.description or ""
            lines.append(f"{env}={default}  # {desc}".rstrip())
        return "\n".join(lines) + "\n"


def identity_from_settings(settings: "Settings | None"):
    """Standalone (no-AF) identity from config. AF replaces this with a JWT-carried identity."""
    from mnemiq.contract import IdentityContext

    return IdentityContext(
        tenant_id=(settings.tenant if settings else None) or "local",
        principal_id=(settings.principal if settings else None) or "local",
        # STRIPPED, because `MNEMIQ_ROLES=analyst, viewer` is what a person writes and it yielded a
        # role named " viewer" -- which matches nothing in the policy, grants nothing, and produces
        # the same accurate-sounding "no tables are available" that #4 was filed about. Every
        # surface resolves roles here, so the CLI, MCP and `/v1` all had it.
        roles=[r.strip() for r in ((settings.roles if settings else "") or "").split(",")
               if r.strip()],
    )
