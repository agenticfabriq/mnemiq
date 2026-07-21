from __future__ import annotations

import json
import os
from dataclasses import dataclass

DEFAULT_MODEL = "openai.gpt-5.5"
DEFAULT_EMBED_MODEL = "openai.text-embedding-3-small"


@dataclass(frozen=True)
class SourceSpec:
    id: str
    kind: str  # "postgres" | "sqlite"
    target: str  # dsn (postgres) or file path (sqlite)
    catalog: str  # DuckDB attach alias AND the object_id prefix in a federated store
    schema: str  # source schema: "public" (postgres) / "main" (sqlite)


@dataclass
class Settings:
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    pg_dsn: str | None
    acme_data_dir: str | None
    embed_model: str | None = None
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    authz_path: str | None = None
    sources_path: str | None = None
    control_dsn: str | None = None
    source_id: str | None = None
    store_path: str | None = None
    default_mode: str | None = None
    write_enabled: bool = False
    verify: bool = False
    verify_override: str | None = None  # raw MNEMIQ_VERIFY: None=mode defaults, "0"=force off, "1"=force full
    verify_threshold: float = 0.5
    verify_sanity: bool = True
    verify_grounding: bool = False  # measured EX cost > sanity; opt-in dial (see plan-27 M1)
    verify_judge: bool = False
    verify_base_url: str | None = None
    verify_model: str | None = None
    verify_api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.llm_model:
            self.llm_model = DEFAULT_MODEL
        if not self.embed_model:
            self.embed_model = DEFAULT_EMBED_MODEL
        if not self.source_id:
            self.source_id = "acme"
        if not self.store_path:
            self.store_path = "mnemiq.duckdb"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_base_url=os.getenv("MNEMIQ_LLM_BASE_URL"),
            llm_api_key=os.getenv("MNEMIQ_LLM_API_KEY"),
            llm_model=os.getenv("MNEMIQ_LLM_MODEL"),
            pg_dsn=os.getenv("MNEMIQ_PG_DSN"),
            acme_data_dir=os.getenv("MNEMIQ_ACME_DATA_DIR"),
            embed_model=os.getenv("MNEMIQ_EMBED_MODEL"),
            embed_base_url=os.getenv("MNEMIQ_EMBED_BASE_URL"),
            embed_api_key=os.getenv("MNEMIQ_EMBED_API_KEY"),
            authz_path=os.getenv("MNEMIQ_AUTHZ_PATH"),
            sources_path=os.getenv("MNEMIQ_SOURCES_PATH"),
            control_dsn=os.getenv("MNEMIQ_CONTROL_DSN"),
            source_id=os.getenv("MNEMIQ_SOURCE_ID"),
            store_path=os.getenv("MNEMIQ_STORE_PATH"),
            default_mode=os.getenv("MNEMIQ_MODE"),
            write_enabled=os.getenv("MNEMIQ_WRITE_ENABLED") == "1",
            verify=os.getenv("MNEMIQ_VERIFY") == "1",
            verify_override=os.getenv("MNEMIQ_VERIFY"),
            verify_threshold=float(os.getenv("MNEMIQ_VERIFY_THRESHOLD", "0.5")),
            verify_sanity=os.getenv("MNEMIQ_VERIFY_SANITY", "1") == "1",
            verify_grounding=os.getenv("MNEMIQ_VERIFY_GROUNDING", "0") == "1",
            verify_judge=os.getenv("MNEMIQ_VERIFY_JUDGE", "0") == "1",
            verify_base_url=os.getenv("MNEMIQ_VERIFY_BASE_URL"),
            verify_model=os.getenv("MNEMIQ_VERIFY_MODEL"),
            verify_api_key=os.getenv("MNEMIQ_VERIFY_API_KEY"),
        )

    def embed_endpoint(self) -> tuple[str | None, str | None]:
        """The endpoint embeddings use. Defaults to the chat endpoint, so today is unchanged;
        set MNEMIQ_EMBED_BASE_URL/KEY to hold embeddings on a separate (e.g. hosted) endpoint
        while chat/generation points at a local model."""
        return (self.embed_base_url or self.llm_base_url, self.embed_api_key or self.llm_api_key)

    def verify_endpoint(self) -> tuple[str | None, str | None]:
        """The endpoint the verifier's judge uses. Defaults to the chat endpoint; set
        MNEMIQ_VERIFY_BASE_URL/KEY to run the judge on a separate (e.g. local) model."""
        return (self.verify_base_url or self.llm_base_url, self.verify_api_key or self.llm_api_key)

    def source_specs(self) -> list["SourceSpec"]:
        """Resolved source list. A manifest wins; otherwise synthesize the single legacy
        source from source_id + pg_dsn (today's behavior -> exactly one spec)."""
        if self.sources_path:
            with open(self.sources_path) as fh:
                raw = json.load(fh)
            return [SourceSpec(id=d["id"], kind=d["kind"], target=d["target"],
                               catalog=d["catalog"], schema=d["schema"]) for d in raw]
        return [SourceSpec(id=self.source_id or "acme", kind="postgres",
                           target=self.pg_dsn or "", catalog="src", schema="public")]
