from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from mnemiq.generate.prompts import system_prompt, user_prompt
from mnemiq.semantic.retrieval import ContextPacket

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Constrained decoding (MNEMIQ_GUIDED_SQL=1): force the reply to be a JSON object with a NON-EMPTY
# sql string, so an over-deferring model literally cannot return {"sql": null}. Measured worth: on
# Qwen-14B this turned a 100% deferral rate into real attempts, because the model was fencing its
# JSON in ```json blocks and the parser saw no SQL.
_GUIDED_SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string", "minLength": 1},
        "reason": {"type": "string"},
        # M35. Declared so a guided reply CAN carry it -- but `_guided_extra_body` STRIPS it from
        # the schema it sends unless the guard that reads it is on, because a property the prompt
        # does not ask for would only invite the model to fill a field nothing reads.
        # How strictly a backend confines a reply to
        # the declared properties is the backend's business and this repo pins no version, so the
        # claim here is only the safe half: a property named in the schema is one the model is
        # asked for and permitted, and one omitted is at best not asked for. That was enough to
        # leave the undefined-term guard inert on the local-model deployments guided mode exists
        # for, in a way no test over the parser could see -- the parser handles the field
        # correctly and simply never receives it.
        "assumed_terms": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sql"],
}


def _guided_extra_body(guided_sql: bool, declare_assumed_terms: bool = False) -> dict | None:
    """The request field that constrains the reply to `_GUIDED_SQL_SCHEMA`.

    `response_format`, not vLLM's `guided_json`. Both were measured against both backends:
    guided_json is a vLLM extension and the OpenAI-compatible endpoint rejects it outright
    with `Unknown parameter` -- a 400, so the request FAILS rather than degrading. This flag
    was therefore not merely inert on the frontier model, it was unusable, and the comment
    here used to claim such endpoints would "ignore" it.

    `strict` is deliberately omitted. Strict mode additionally requires
    `additionalProperties: false` and every property in `required`, which `reason` (optional
    by design) violates -- sending it is another 400. Without strict the schema is honoured
    by vLLM's structured-output engine and is best-effort on the provider side, so the
    non-empty guarantee is firm where it was always needed and advisory where the model
    was not deferring anyway.
    """
    if not guided_sql:
        return None
    schema = _GUIDED_SQL_SCHEMA
    if not declare_assumed_terms:
        # A property the prompt does not ask for has no business in the grammar either: it would
        # invite the model to fill a field nothing reads.
        schema = {**schema, "properties": {k: v for k, v in schema["properties"].items()
                                           if k != "assumed_terms"}}
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "sql_proposal", "schema": schema},
        }
    }


@dataclass
class SqlProposal:
    sql: str | None  # None is a deferral, not a failure
    reason: str = ""
    # M35: business terms the model had to ASSUME a meaning for. It rides in the reply that
    # produces the SQL rather than in a call of its own -- the model has already read the question
    # and the packet, so a second round trip would ask the same model about the same text.
    #
    # A DECLARATION, not a decision. `undefined_terms.ungrounded_terms` checks each against the
    # certified set and the planner refuses; asking the model to refuse when unsure is asking the
    # thing that invented the derivation to notice it invented it, which it does 2 times in 6.
    assumed_terms: list[str] = field(default_factory=list)


def _parse(raw: str) -> SqlProposal:
    """Read the structured channel. A reply we cannot read is a deferral, never an exception."""
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return SqlProposal(sql=None, reason="the model did not answer on the JSON channel")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return SqlProposal(sql=None, reason="the model's JSON was malformed")
    if not isinstance(payload, dict):
        return SqlProposal(sql=None, reason="the model's JSON was not an object")

    sql = payload.get("sql")
    reason = payload.get("reason")
    # Absent means "declared nothing", never "unknown": every existing prompt and every test double
    # omits the key, and a missing field that deferred would break the engine rather than guard it.
    assumed = payload.get("assumed_terms")
    return SqlProposal(
        sql=sql.strip() if isinstance(sql, str) and sql.strip() else None,
        reason=reason if isinstance(reason, str) else "",
        assumed_terms=[t.strip() for t in assumed if isinstance(t, str) and t.strip()]
        if isinstance(assumed, list)
        else [],
    )


class Generator(Protocol):
    def propose(self, packet: ContextPacket, feedback: str | None = None) -> SqlProposal: ...


class LLMGenerator:
    def __init__(self, client, max_tokens: int = 4000, dialect: str = "duckdb",
                 guided_sql: bool = False, assertive: bool = False,
                 declare_assumed_terms: bool = False) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._dialect = dialect
        self._guided_sql = guided_sql
        self._assertive = assertive
        self._declare_assumed_terms = declare_assumed_terms

    def propose(
        self, packet: ContextPacket, feedback: str | None = None, strategy: str | None = None
    ) -> SqlProposal:
        extra = _guided_extra_body(self._guided_sql, self._declare_assumed_terms)
        raw = self._client.complete(
            system_prompt(dialect=self._dialect, strategy=strategy, assertive=self._assertive,
                          declare_assumed_terms=self._declare_assumed_terms),
            user_prompt(packet, feedback),
            max_tokens=self._max_tokens,
            **({"extra_body": extra} if extra else {}),
        )
        return _parse(raw)


class StrategyGenerator:
    """Binds one strategy onto a generator, behind the plain two-arg Generator protocol --
    plan_query never learns strategies exist."""

    def __init__(self, inner, strategy: str) -> None:
        self._inner = inner
        self._strategy = strategy

    def propose(self, packet: ContextPacket, feedback: str | None = None) -> SqlProposal:
        return self._inner.propose(packet, feedback, strategy=self._strategy)


class FakeGenerator:
    """Replays scripted model replies through the real parser. Zero tokens."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str | None] = []
        self.strategies: list[str | None] = []

    def propose(
        self, packet: ContextPacket, feedback: str | None = None, strategy: str | None = None
    ) -> SqlProposal:
        self.calls.append(feedback)
        self.strategies.append(strategy)
        if not self._replies:
            return SqlProposal(sql=None, reason="no more scripted replies")
        return _parse(self._replies.pop(0))
