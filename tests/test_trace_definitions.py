"""**M138.** An answer's trace names the definitions its prompt held.

`build_trace` set `definitions_used=[]` with "the glossary lands in Plan 08"; the glossary landed --
`select_definitions` fills `packet.definitions`, the prompt carries them, the fit leaves them alone
-- and every trace still said no definition was used, including answers grounded on one. The field
is in the published contract, so a consumer learned something false rather than nothing.
"""

from dataclasses import replace

from mnemiq.contract import Definition
from tests.test_agent import _GRANTS, _IDENTITY, _agent, _packet, _snapshot

_PAYMENT_DATE = Definition(
    id="fspay:policy:business_date", term="business date", domain="payments",
    definition="the business date of a claim is its settlement date, not its entry date",
    bound_objects=["claim"])


def _trace(packet):
    answer = _agent(['{"sql": "SELECT n FROM claim"}']).answer(
        packet, _snapshot(), _GRANTS, _IDENTITY)
    assert answer.trace is not None, f"setup: the answer was not given: {answer.answer}"
    return answer.trace


def test_a_trace_names_the_definitions_the_prompt_held():
    trace = _trace(replace(_packet(), definitions=[_PAYMENT_DATE]))
    assert trace.definitions_used == ["fspay:policy:business_date"]


def test_a_trace_with_no_definitions_in_the_prompt_names_none():
    assert _trace(_packet()).definitions_used == []
