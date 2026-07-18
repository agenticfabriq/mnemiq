from mnemiq.contract import Column, Snapshot
from mnemiq.enrichment.refresh import catalog_diff


class _FakeAdapter:
    def __init__(self, cols):  # cols: list[(table, col, type)]
        self._cols = cols

    def introspect(self):
        return sorted({t for t, _, _ in self._cols})

    def list_columns(self):
        return self._cols


def _snap(cols):  # cols: list[(table, col)]
    return Snapshot(version="v1", source_id="acme", created_at="t",
                    columns=[Column(id=f"{t}.{c}", object_id=t, name=c) for t, c in cols])


def test_catalog_diff_detects_added_changed_dropped():
    snap = _snap([("person", "id"), ("claim", "id"), ("old_t", "x")])
    adapter = _FakeAdapter([("person", "id", "INT"),          # unchanged
                            ("claim", "id", "INT"), ("claim", "amount", "INT"),  # changed (+amount)
                            ("new_t", "y", "INT")])            # added; old_t dropped
    diff = catalog_diff(adapter, snap)
    assert diff.added == ["new_t"]
    assert diff.changed == ["claim"]
    assert diff.dropped == ["old_t"]
    assert diff.unchanged == ["person"]
    assert diff.has_changes is True


def test_catalog_diff_no_changes():
    snap = _snap([("person", "id")])
    adapter = _FakeAdapter([("person", "id", "INT")])
    diff = catalog_diff(adapter, snap)
    assert diff.has_changes is False
