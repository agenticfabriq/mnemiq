"""M42: a certified record supplies the meaning it states, and leaves alone what it does not.

`apply_certified` wrote all five overlaid fields from every column record, so a record that
omitted `pii_level` set it to None -- and None is what `_qualifies` and `build_access_policy` both
read as "not sensitive". Measured before the fix: a `phi` column plus an ATTESTED record with no
`pii_level` key ended `None`, its values qualified for the prompt's value index, and an uncleared
identity's policy denied nothing. The owner's decision (2026-10-13): absent means unchanged, for
every overlaid field. Records are built from wire dicts, through `CertifiedRecord.model_validate`,
because which keys a record carried is exactly what the fix reads.
"""

from __future__ import annotations

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import CodedValue, Column, Snapshot
from mnemiq.contract.records import CertifiedRecord
from mnemiq.contract.semantic import CodeScheme
from mnemiq.enrichment.certified import apply_certified
from mnemiq.semantic.values import _qualifies
from mnemiq.sql.policy import build_access_policy

_LOCAL = {
    "description": "local guess",
    "semantic_type": "diagnosis_code",
    "pii_level": "phi",
    "coded_values": [CodedValue(code="A1", meaning="local meaning", source="lookup")],
    "code_scheme": CodeScheme(id="local", label="Local scheme"),
}
_STATED = {
    "description": "certified meaning",
    "semantic_type": "icd10_code",
    "pii_level": "pii",
    "coded_values": [{"code": "A1", "meaning": "certified meaning"}],
    "code_scheme": {"id": "icd10", "label": "ICD-10"},
}


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id="patient.dx_code", object_id="patient", name="dx_code", data_type="varchar",
               distinct_count=12, **_LOCAL)])


def _record(**payload) -> CertifiedRecord:
    return CertifiedRecord.model_validate({
        "envelope": {"object_type": "column", "object_id": "patient.dx_code", "version": "h1",
                     "source_system": "verity",
                     "provenance": {"status": "certified", "certifier": "steward@acme.example"}},
        "payload": {"id": "patient.dx_code", "object_id": "patient", "name": "dx_code", **payload},
    })


def _overlaid(**payload) -> Column:
    return apply_certified(_snapshot(), [_record(**payload)]).columns[0]


def test_an_attested_record_that_omits_pii_level_leaves_the_column_sensitive():
    """The finding end to end: the level, the value index's gate, and the policy built from it."""
    column = _overlaid(description="certified meaning")
    assert column.pii_level == "phi"
    assert not _qualifies(column, max_distinct=200)
    policy = build_access_policy(Snapshot(version="v1", source_id="s", created_at="t",
                                          columns=[column]),
                                 GrantSet(frozenset({"patient"})))
    assert ("patient", "dx_code") in policy.denied


def test_an_explicit_null_level_is_no_statement_either():
    """`none` is the vocabulary's word for not personal; a null is the value consumers read as
    harmless, so it must not stand for one."""
    assert _overlaid(pii_level=None).pii_level == "phi"


@pytest.mark.parametrize("field", sorted(_LOCAL))
def test_a_field_the_record_omits_keeps_its_local_value(field):
    others = {k: v for k, v in _STATED.items() if k != field}
    assert getattr(_overlaid(**others), field) == _LOCAL[field]


@pytest.mark.parametrize("field", sorted(_LOCAL))
def test_a_field_the_record_states_is_overlaid(field):
    """The control: statements still win, so the test above is about silence, not about the
    overlay having stopped."""
    column = _overlaid(**{field: _STATED[field]})
    assert getattr(column, field) != _LOCAL[field]
