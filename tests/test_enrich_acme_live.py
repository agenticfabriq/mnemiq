import os
import re

import pytest

from acme_dsn import acme_dsn, requires_acme

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.config import Settings
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.llm.client import LLMClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live_llm,
    pytest.mark.skipif(not os.getenv("MNEMIQ_LLM_API_KEY"), reason="no live LLM configured"),
    requires_acme,
]

_DSN = acme_dsn()


def test_live_enrichment_gives_acme_its_meaning():
    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))
    structural = enrich_structural(adapter, "acme")
    enriched = enrich_semantic(structural, LLMEnricher(LLMClient(Settings.from_env())))

    described = [c for c in enriched.columns if c.description]
    assert len(described) > len(enriched.columns) / 2, "most columns should be documented"

    # the codes the structural pass observed now have meanings
    fireplace = next(c for c in enriched.columns if c.id == "fireclaim.fireplace")
    assert all(cv.meaning for cv in fireplace.coded_values)

    # identifiers are recognized as references, and the model stayed inside the vocabulary
    typed_ids = [c for c in enriched.columns if c.name.endswith("_identifier") and c.semantic_type]
    assert any(c.semantic_type == "identifier" for c in typed_ids)

    assert enriched.version != structural.version


def test_live_enrichment_carries_no_personal_data():
    """The snapshot is a searchable artifact. Real people must not be in it."""
    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))
    enriched = enrich_semantic(
        enrich_structural(adapter, "acme"), LLMEnricher(LLMClient(Settings.from_env()))
    )

    # ACME's person table holds these names; none may survive into the snapshot.
    # Whole words only -- "primary street address" is not a person called Mary.
    blob = enriched.model_dump_json().lower()
    for name in ("alice", "bob", "mary"):
        assert not re.search(rf"\b{name}\b", blob), f"a real person's name reached the snapshot: {name}"

    # no column classified as personal data may carry its observed values
    for column in enriched.columns:
        if column.pii_level in {"pii", "phi"}:
            assert column.coded_values == [], f"{column.id} carries personal values"
