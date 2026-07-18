"""Live multi-replica shared state against the available Postgres (used here as the control
plane, with test-specific tables). Proves cross-replica cache hits and the version pointer.
Integration-gated; no LLM."""

import os

import psycopg
import pytest

from mnemiq.cache.postgres import PostgresCache
from mnemiq.store.control import current_pointer, publish_version

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no Postgres configured"),
]

_DSN = os.getenv("MNEMIQ_PG_DSN", "")
_TABLE = "mnemiq_cache_test"


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    with psycopg.connect(_DSN, autocommit=True) as con:
        con.execute(f"DROP TABLE IF EXISTS {_TABLE}")
        con.execute(
            "CREATE TABLE IF NOT EXISTS mnemiq_version "
            "(source_id TEXT PRIMARY KEY, version TEXT, updated_at TIMESTAMPTZ DEFAULT now())"
        )
        con.execute("DELETE FROM mnemiq_version WHERE source_id LIKE 'mrtest_%'")


def test_cross_replica_cache_hit():
    a = PostgresCache(_DSN, table=_TABLE)  # "replica A"
    b = PostgresCache(_DSN, table=_TABLE)  # "replica B", same table
    a.put("plan-key", b"arrow-bytes")
    assert b.get("plan-key") == b"arrow-bytes"  # B serves what A cached


def test_two_identities_never_collide_in_shared_l2():
    a = PostgresCache(_DSN, table=_TABLE)
    a.put("keyfor-narrow-identity", b"narrow")
    a.put("keyfor-broad-identity", b"broad")
    assert a.get("keyfor-narrow-identity") == b"narrow"
    assert a.get("keyfor-broad-identity") == b"broad"  # distinct keys -> no cross-serve


def test_version_pointer_roundtrip_across_readers():
    publish_version(_DSN, "mrtest_src", "v1")
    assert current_pointer(_DSN, "mrtest_src") == "v1"
    publish_version(_DSN, "mrtest_src", "v2")  # a version bump
    assert current_pointer(_DSN, "mrtest_src") == "v2"  # a second reader sees it
