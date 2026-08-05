"""The one place the LLM-facing component kit is assembled.

build_runtime (product) and eval's build_engine wrap this same kit with different policy:
cache tier, verifier, authz, and index provisioning stay wrapper-owned on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from mnemiq.agent.synthesize import LLMSynthesizer
from mnemiq.config import Settings
from mnemiq.execute.select import LLMSelector
from mnemiq.generate.correct import LLMCorrector
from mnemiq.generate.generator import LLMGenerator
from mnemiq.llm.client import LLMClient
from mnemiq.semantic.values import ValueIndex


@dataclass(frozen=True)
class Components:
    client: LLMClient
    generator: LLMGenerator
    synthesizer: LLMSynthesizer
    corrector: LLMCorrector
    values: ValueIndex
    selector: LLMSelector


def build_components(settings: Settings, adapter, con) -> Components:
    client = LLMClient(settings)
    # Generate in the dialect the source executes, so generation, parsing, and execution stay
    # on one dialect and no cross-dialect transpile gap can bite.
    generator = LLMGenerator(client, dialect=getattr(adapter, "dialect", "duckdb"),
                             guided_sql=settings.guided_sql, assertive=settings.assertive_sql)
    return Components(client=client, generator=generator, synthesizer=LLMSynthesizer(client, markdown=settings.answer_markdown),
                      corrector=LLMCorrector(client), values=ValueIndex(con),
                      selector=LLMSelector(client))
