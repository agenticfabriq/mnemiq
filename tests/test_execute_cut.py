"""The mark an answer cut to fit one response carries on its own table (`mnemiq.execute.cut`)."""

from __future__ import annotations

import pyarrow as pa

from mnemiq.cache.store import from_ipc, to_ipc
from mnemiq.execute.cut import mark_sent, sent_of
from mnemiq.execute.render import render_result

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


def test_a_rendered_cut_table_says_it_is_not_the_whole():
    """Every model that reads the rows (the writer, the verifier, the selector) reads that they are not all of them."""
    rendered = render_result(mark_sent(TABLE, 2))
    assert "do not report a total or a count" in rendered
    assert "do not report a total" not in render_result(TABLE)
