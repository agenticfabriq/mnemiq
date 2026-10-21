"""A cancelled Oracle call is a timeout, not the statement's failure (M187): `_is_timeout`
reads the driver's error text, and ORA-01013, the spelling a call timeout takes in thick mode
and under a server-side cancel by the driver's own error list, was not among the spellings it
matched, so it reached the runner as a plain source fault while DPY-4024 (thin mode) and
ORA-03156 were reported as "query timed out". No database here: the matcher reads text."""

from __future__ import annotations

import pytest

oracledb = pytest.importorskip("oracledb")

from mnemiq.adapters.oracle import _is_timeout  # noqa: E402


@pytest.mark.parametrize("text", [
    "ORA-01013: user requested cancel of current operation",
    "DPY-4024: call timeout of 300 ms exceeded",
    "DPY-4024: reworded by a later driver",   # the code, not the wording (the gate)
    "ORA-03156: the statement timed out",   # the code is what the matcher reads
    "DPY-4011: the database or network closed the connection",
])
def test_a_bounded_calls_end_is_a_timeout_under_each_spelling(text):
    assert _is_timeout(oracledb.DatabaseError(text)), text


@pytest.mark.parametrize("text", [
    "ORA-00942: table or view does not exist",
    "ORA-01017: invalid username/password; logon denied",
    "DPY-4033: the database closed the connection because the session was idle",
])
def test_a_fault_that_is_no_timeout_is_not_read_as_one(text):
    assert not _is_timeout(oracledb.DatabaseError(text)), text
