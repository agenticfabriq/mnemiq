"""The result judge reads the context the generator was given: certified meaning and earlier turns.

M132. Handed the cards alone, the judge declined three of six fs payments answers that were the
certified expression verbatim (0.12 to 0.25 against the 0.5 cutoff) while passing all six of the
bare arm's wrong answers (0.62 to 0.95), because it graded each against its own reading of words
like "net" and "gross". Given the certified block, the six right answers scored 0.97 to 1.0, and
nine of ten distinct wrong queries stayed under the cutoff on every read.

M134. Without the earlier turns, the judge declined three of five right follow-ups ("and by
channel?", "which one is the largest?") at 0.15 to 0.45; given them, all five scored 0.97 to 1.0,
and five wrong follow-ups 0.0 to 0.10.
"""

import dataclasses
from types import SimpleNamespace

import pyarrow as pa
import pytest

from mnemiq.contract import Definition, Dimension, Example, HistoryTurn, MeasureExpr, Metric
from mnemiq.generate.prompts import user_prompt
from mnemiq.semantic.retrieval import ContextPacket, ResolvedConcept, RetrievedCard
from mnemiq.sql.verdict import Approved
from mnemiq.verify.verifier import Verifier

_CARD = "TABLE payment_transaction\n- payment_amount numeric\n- payment_channel text"
_NET = "sum(payment_transaction.payment_amount) filter (where payment_transaction.is_latest)"
_SQL = f"SELECT {_NET} AS net_payment_volume FROM payment_transaction"


def _meaning() -> dict:
    return {
        "definitions": [
            Definition(id="d1", term="net volume", domain="payments",
                       definition="the amount of each payment's latest version, before tips"),
        ],
        "metrics": [
            Metric(id="net_payment_volume", label="Net Payment Volume", status="certified",
                   owner="o", grain="payment",
                   measure=MeasureExpr(expr=_NET, source="payment_transaction"),
                   time_dimension="payment_transaction_date"),
        ],
        "dimensions": [
            Dimension(id="payment_channel", label="Payment Channel",
                      source="payment_transaction", expr="payment_channel"),
        ],
        "concepts": [
            ResolvedConcept("payment_transaction.payment_channel", "Channels", "MOB", "Mobile"),
        ],
    }


_TURN = HistoryTurn(question="What is the net payment volume?", sql=_SQL,
                    columns=["net_payment_volume"], rows=[[1808030.48]], grant_fingerprint="f")


def _packet(**meaning) -> ContextPacket:
    packet = ContextPacket(
        question="what is the net payment volume?",
        cards=[RetrievedCard(object_id="payment_transaction", card=_CARD, score=1.0)],
        grant_fingerprint="f", enrichment_version="v")
    for name, value in meaning.items():
        setattr(packet, name, value)
    return packet


def _judged_schema(packet: ContextPacket, protocol: str = "read") -> str:
    seen = {}

    class _Scorer:
        def score(self, question, schema, sql, preview):
            seen["schema"] = schema
            return 0.9

    class _Reader:  # no usable `score`, so this case proves the `read` door was taken
        def score(self, question, schema, sql, preview):
            raise AssertionError("a judge offering `read` must be read, not scored")

        def read(self, question, schema, sql, preview):
            seen["schema"] = schema
            return SimpleNamespace(score=0.9, fell_open=False, reason="ok")

    judge = _Reader() if protocol == "read" else _Scorer()
    Verifier(sanity=False, judge=judge).verify(
        packet, Approved(plan_sql=_SQL, target_sql=_SQL), pa.table({"net_payment_volume": [1.5]}))
    return seen["schema"]


@pytest.mark.parametrize("protocol", ["read", "score"])
def test_the_judge_reads_the_certified_meaning(protocol):
    schema = _judged_schema(_packet(**_meaning()), protocol)
    assert _CARD in schema
    assert _NET in schema
    assert "the amount of each payment's latest version, before tips" in schema
    assert "Payment Channel = payment_channel on payment_transaction" in schema
    assert "MOB = Mobile" in schema


def test_the_judge_reads_every_meaning_line_the_generator_reads():
    """Against the generator's own prompt, not against the shared renderer: a section that
    `user_prompt` renders outside `meaning_sections`, from a field this packet fills, reaches the
    generator and not the judge. A section from a NEW field would sit empty here; the field guard
    below is what catches that one."""
    packet = _packet(**_meaning())
    lines = user_prompt(packet).splitlines()
    generator_meaning = [line for line in lines[1:lines.index("TABLES:")] if line.strip()]
    assert len(generator_meaning) == 8  # four sections, a header and one item each
    judged = _judged_schema(packet).splitlines()
    assert [line for line in generator_meaning if line not in judged] == []


def test_a_packet_with_no_certified_meaning_is_judged_as_before():
    """The 0.5 threshold was measured on packets like this one, so its text must not move."""
    assert _judged_schema(_packet()) == _CARD


@pytest.mark.parametrize("protocol", ["read", "score"])
def test_the_judge_reads_the_earlier_turns(protocol):
    schema = _judged_schema(_packet(history=[_TURN]), protocol)
    assert _CARD in schema
    assert "Q: What is the net payment volume?" in schema
    assert f"SQL: {_SQL}" in schema
    assert "1808030.48" in schema


def test_the_judge_reads_every_history_line_the_generator_reads():
    packet = _packet(history=[_TURN])
    lines = user_prompt(packet).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("EARLIER IN THIS CONVERSATION"))
    generator_history = [line for line in lines[start:lines.index("TABLES:")] if line.strip()]
    assert len(generator_history) == 5  # header, Q, SQL, RESULT, one row
    judged = _judged_schema(packet).splitlines()
    assert [line for line in generator_history if line not in judged] == []


def test_every_packet_field_is_judged_or_left_out_on_purpose():
    """Derived from the struct: a field added to `ContextPacket` fails this until it is placed on
    one side, and the placement is checked both ways -- a judged field, filled on its own, changes
    what the judge reads, and a field left out does not (M132 and M134 one field over)."""
    judged = {"cards", "definitions", "metrics", "dimensions", "concepts", "history"}
    left_out = {
        "question": "a different question",  # handed to the judge on its own
        "grant_fingerprint": "another-boundary",  # bookkeeping, not prompt text
        "enrichment_version": "v2",
        "descriptions": (1, 2),
        # How to write SQL over these tables, not what a term means or what was asked before.
        "examples": [Example(question="q", sql="SELECT 1", object_id="payment_transaction")],
    }
    assert {f.name for f in dataclasses.fields(ContextPacket)} == judged | set(left_out)
    filled = {**_meaning(), "history": [_TURN]}
    assert set(filled) == judged - {"cards"}
    unread = [name for name, value in filled.items()
              if _judged_schema(_packet(**{name: value})) == _CARD]
    assert unread == []
    leaked = [name for name, value in left_out.items()
              if _judged_schema(_packet(**{name: value})) != _CARD]
    assert leaked == []
