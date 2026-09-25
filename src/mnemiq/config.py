from __future__ import annotations

import json
from dataclasses import dataclass

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

_SECRET_HINTS = ("api_key", "dsn", "password", "secret")  # fields whose value is never printed in .env.example

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
                     "this machine or network (see assert_local_only for exactly what this checks "
                     "and what it deliberately does not)",
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
    ack_advisories: str | None = Field(default=None, description="boot advisories to log at INFO instead of WARNING, comma-separated as <advisory>:<verdict> (e.g. read-only-basis:unverifiable); a CHANGED verdict still warns")
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
    card_style: Literal["cards", "ddl"] = Field(default="cards",
                            description="schema card form for the GENERATOR: cards|ddl. "
                                        "`ddl` renders CREATE TABLE for a model fine-tuned "
                                        "on DDL; the retrieval index always embeds `cards`.")
    guided_sql: bool = Field(default=False, description="constrained decoding: force non-empty sql (response_format json_schema; works on vLLM and OpenAI-compatible endpoints)")
    assertive_sql: bool = Field(default=False, description="assertive prompt: attempt an answer instead of deferring")
    # M35, off by default: withdrawn on its own pre-registered criterion. Beacon's answerable band
    # deferred 12 of 24 against a prior of 0 in 144, where the threshold named in advance was 2-3%.
    # See `plan_query` for what the number decomposes into and what reviving it would take.
    guard_undefined_terms: bool = Field(default=False, description="M35: refuse a declared business term with no certified definition (withdrawn -- see plan_query)")
    # M109. Off: the pre-registered paired measurement's no-harm bar on BIRD (at most one
    # correct answer lost) was not met on the final code, though the repairs on a multi-fact
    # schema cleared theirs. The default is pinned by a test.
    guard_fanout: bool = Field(default=False, description="M109: refuse an aggregate over rows a join has multiplied, as a repairable refusal (key uniqueness from column profiles)")
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
    llm_seed: int | None = Field(default=None, description="sampling seed forwarded to the provider; honoured by vLLM, IGNORED by the hosted endpoint (accepted, no system_fingerprint, output still varies) -- so it buys reproducibility on the local path only")
    retrieval_k: int = Field(default=DEFAULT_RETRIEVAL_K, ge=1, description="schema cards retrieved into the generation packet")
    definition_index_max_concepts: int = Field(default=500, ge=0,
        description="per-scheme concept cap for the enrich-time definition-grounding index")
    binding_suggestions_path: str | None = Field(default=None,
        description="where enrich writes ontology binding-suggestions (defaults beside the store)")
    principal: str | None = Field(default=None, description="MCP standalone identity: principal id")
    roles: str | None = Field(default=None, description="MCP standalone identity: comma-separated roles")
    tenant: str | None = Field(default=None, description="MCP standalone identity: tenant id")
    # --- verifier ---
    verify_override: str | None = Field(default=None, validation_alias="MNEMIQ_VERIFY",
                                        description="verify override: unset=mode defaults, 0=force off, 1=force full")
    verify_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="judge-confidence defer cutoff")
    verify_sanity: bool = Field(default=True, description="deterministic empty/null defer layer")
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

    def assert_local_only(self) -> None:
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
        import ipaddress
        import sys
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
            try:
                host = urlparse(url).hostname
            except ValueError as exc:
                # `urlparse` raises for a malformed IPv6 host literal (e.g. an unterminated
                # `[`) BEFORE `.hostname` is reached -- a different failure point than the
                # `host is None` case below, and one an earlier version of this method let
                # propagate uncaught, discarding every offender already collected for a field
                # visited before this one in loop order.
                offenders.append(f"{name}={url!r} (not a parseable URL: {exc})")
                continue
            if host is None:
                offenders.append(f"{name}={url!r} (no host could be parsed)")
                continue
            if host in ("localhost", "localhost.localdomain"):
                checked.append(name)
                continue
            try:
                addr = ipaddress.ip_address(host)
            except ValueError:
                offenders.append(
                    f"{name}={url!r} (host {host!r} is a DNS name; this check does not "
                    "resolve names, so it cannot confirm one stays on this network)"
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
                offenders.append(f"{name}={url!r} (host {host} is publicly routable)")
                continue
            checked.append(name)
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
        cleared = ", ".join(checked) if checked else "(no chat/embedding/judge/verity endpoint configured)"
        print(
            f"MNEMIQ_LOCAL_ONLY verified: {cleared} -- pg_dsn/control_dsn are not checked here",
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
                               catalog=d["catalog"], schema=d["schema"]) for d in raw]
        return [SourceSpec(id=self.source_id or "acme", kind="postgres",
                           target=self.pg_dsn or "", catalog="src", schema="public")]

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
