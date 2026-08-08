from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL = "openai.gpt-5.5"
DEFAULT_EMBED_MODEL = "openai.text-embedding-3-small"

_SECRET_HINTS = ("api_key", "dsn", "password", "secret")  # fields whose value is never printed in .env.example


@dataclass(frozen=True)
class SourceSpec:
    id: str
    kind: str  # "postgres" | "sqlite"
    target: str  # dsn (postgres) or file path (sqlite)
    catalog: str  # DuckDB attach alias AND the object_id prefix in a federated store
    schema: str  # source schema: "public" (postgres) / "main" (sqlite)


class Settings(BaseSettings):
    """All engine configuration, one source of truth. `env_prefix="MNEMIQ_"` maps each field to its
    env var (llm_base_url -> MNEMIQ_LLM_BASE_URL), so every historical name is preserved. No env_file:
    Settings() reads os.environ only (scripts `source .env`), which also keeps tests isolated."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMIQ_", extra="ignore", populate_by_name=True)

    # --- connection ---
    llm_base_url: str | None = Field(default=None, description="chat/generation OpenAI-compat base URL")
    llm_api_key: str | None = Field(default=None, description="chat/generation API key")
    llm_model: str | None = Field(default=None, description="chat/generation model id")
    pg_dsn: str | None = Field(default=None, description="source Postgres DSN")
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
    guided_sql: bool = Field(default=False, description="constrained decoding: force non-empty sql (vLLM guided_json)")
    assertive_sql: bool = Field(default=False, description="assertive prompt: attempt an answer instead of deferring")
    enrich_facts: bool = Field(default=False, description="eval: table-facts enrichment phase (plan-20, default off)")
    enrich_examples: bool = Field(default=False, description="eval: verified-example enrichment phase (plan-20, default off)")
    dictionary_path: str | None = Field(default=None, description="operator code data-dictionary JSON path (grounds code meanings)")
    ontology_records_path: str | None = Field(default=None, description="ontology records JSON path (binds code schemes, grounds bare codes)")
    verity_records_url: str | None = Field(default=None, description="Verity GET /api/semantic/records endpoint (governed certified records)")
    verity_watermark_path: str | None = Field(default=None, description="path to the Verity /open incremental-sync watermark sidecar JSON (defaults beside the store)")
    verity_page_size: int = Field(default=500, ge=0, description="Verity /open page size sent as ?limit= (0 = unbounded full dump)")
    verity_full_resync_after_secs: int = Field(default=86400, ge=0, description="how stale the locally merged certified set may get before it is re-pulled in full; a withdrawal is invisible to an incremental delta, so this window is the only thing that removes one locally (0 = always full)")
    verity_token_url: str | None = Field(default=None, description="Keycloak POST .../realms/<realm>/protocol/openid-connect/token endpoint (OAuth2 client-credentials grant)")
    verity_client_id: str | None = Field(default=None, description="Verity service client id (created in the workbench API Credentials page)")
    verity_client_secret: str | None = Field(default=None, description="Verity service client secret; presented only to the token endpoint, never to the records endpoint")
    retrieval_k: int = Field(default=12, ge=1, description="schema cards retrieved into the generation packet")
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
        roles=[r for r in ((settings.roles if settings else "") or "").split(",") if r],
    )
