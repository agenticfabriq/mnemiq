"""The mark an answer cut to fit one response carries on its own table (`mnemiq.execute.cut`)."""

from __future__ import annotations

import pyarrow as pa

from mnemiq.cache.store import from_ipc, to_ipc
from mnemiq.execute.cut import mark_sent, sent_of

TABLE = pa.table({"region": ["West", "East"], "policies": [12, 7]})


def test_a_marked_table_reads_its_count():
    assert sent_of(mark_sent(TABLE, 92)) == 92
    assert sent_of(TABLE) is None


def test_the_mark_survives_the_result_cache():
    """A cut answer served again from the cache still says it was cut."""
    assert sent_of(from_ipc(to_ipc(mark_sent(TABLE, 92)))) == 92


def test_the_mark_does_not_change_equality():
    """Two candidates' equal results stay equal whether or not one was cut."""
    assert mark_sent(TABLE, 92).equals(TABLE)
