from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "openai.gpt-5-mini"
DEFAULT_EMBED_MODEL = "openai.text-embedding-3-small"


@dataclass
class Settings:
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    pg_dsn: str | None
    acme_data_dir: str | None
    embed_model: str | None = None
    authz_path: str | None = None

    def __post_init__(self) -> None:
        if not self.llm_model:
            self.llm_model = DEFAULT_MODEL
        if not self.embed_model:
            self.embed_model = DEFAULT_EMBED_MODEL

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_base_url=os.getenv("MNEMIQ_LLM_BASE_URL"),
            llm_api_key=os.getenv("MNEMIQ_LLM_API_KEY"),
            llm_model=os.getenv("MNEMIQ_LLM_MODEL"),
            pg_dsn=os.getenv("MNEMIQ_PG_DSN"),
            acme_data_dir=os.getenv("MNEMIQ_ACME_DATA_DIR"),
            embed_model=os.getenv("MNEMIQ_EMBED_MODEL"),
            authz_path=os.getenv("MNEMIQ_AUTHZ_PATH"),
        )
