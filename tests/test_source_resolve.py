"""The dispatch from a SourceSpec to an adapter, proved without a live database.

Every assertion here monkeypatches the adapter CLASSES and inspects the constructor arguments.
That is deliberate: constructing a real `DuckDBAdapter` runs `ATTACH`, so a test that built one
could only be written against a running Postgres, and the thing under test -- which adapter, with
which arguments -- would be the one part not observed. Recording the call makes the decision the
assertion.
"""

from __future__ import annotations

import json

import pytest

from mnemiq.adapters import resolve
from mnemiq.config import Settings, SourceSpec


class _Recorder:
    """Stands in for an adapter class and keeps the kwargs it was constructed with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def recorded(monkeypatch):
    monkeypatch.setattr(resolve, "DuckDBAdapter", _Recorder)
    monkeypatch.setattr(resolve, "OracleAdapter", _Recorder)


def _settings(**kw) -> Settings:
    base = dict(llm_base_url=None, llm_api_key=None, llm_model=None, acme_data_dir=None)
    return Settings(**{**base, **kw})


def _spec(**kw) -> SourceSpec:
    base = dict(id="s", kind="postgres", target="t", catalog="src", schema="public")
    return SourceSpec(**{**base, **kw})


# -- the legacy path must not move ------------------------------------------------------------

def test_the_default_postgres_source_attaches_exactly_as_before(recorded):
    """The no-manifest spec must produce the arguments `DuckDBPostgresAdapter(dsn)` produced.

    The single-source branch used to construct that subclass directly. It only overrides
    `__init__`, and nothing anywhere does an isinstance check on it, so the base class built with
    these arguments is the same adapter -- but "the same" is a claim, so it is pinned here.
    """
    s = _settings(pg_dsn="postgresql://h/db")
    adapter = resolve.adapter_for(s.source_specs()[0], s)
    assert adapter.kwargs == {
        "attach_target": "postgresql://h/db", "attach_type": "POSTGRES",
        "extension": "postgres", "catalog": "src", "table_schema": "public",
        "fk_via_postgres": True, "read_only": True,
    }


def test_read_only_is_threaded_through_rather_than_defaulted(recorded):
    assert resolve.adapter_for(_spec(), _settings(), read_only=False).kwargs["read_only"] is False


# -- the bug this module was written to close --------------------------------------------------

def test_a_one_entry_manifest_is_honoured_not_discarded(recorded, tmp_path):
    """A single-source manifest used to be resolved into a spec and then ignored.

    `build_runtime` federates on two or more specs and took `settings.pg_dsn` on one, so a manifest
    naming one SQLite file connected to Postgres instead and said nothing. The decoy DSN is the
    whole point: if dispatch regressed to reading `pg_dsn`, this attaches the decoy.
    """
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "sqlite", "target": "/data/one.db",
                                     "catalog": "s", "schema": "main"}]))
    s = _settings(sources_path=str(manifest), pg_dsn="postgresql://decoy/decoy")
    specs = s.source_specs()
    assert len(specs) == 1, "precondition: this is the single-source branch"
    kwargs = resolve.adapter_for(specs[0], s).kwargs
    assert kwargs["attach_target"] == "/data/one.db"
    assert kwargs["attach_type"] == "SQLITE"
    assert "decoy" not in json.dumps(kwargs)


def test_a_manifest_schema_reaches_the_adapter(recorded):
    """`DuckDBPostgresAdapter` hardcodes `table_schema="public"`, so a manifest naming another
    Postgres schema was silently attached at `public`. The general constructor takes the spec's."""
    kwargs = resolve.adapter_for(_spec(schema="sales", catalog="warehouse"), _settings()).kwargs
    assert (kwargs["table_schema"], kwargs["catalog"]) == ("sales", "warehouse")


# -- oracle ------------------------------------------------------------------------------------

def test_oracle_takes_credentials_from_settings_and_the_descriptor_from_the_spec(recorded):
    s = _settings(oracle_user="app", oracle_password="pw")
    spec = _spec(kind="oracle", target="db.example.com:1521/PDB1", schema="APP")
    assert resolve.adapter_for(spec, s, read_only=False).kwargs == {
        "dsn": "db.example.com:1521/PDB1", "user": "app", "password": "pw",
        "schema": "APP", "read_only": False,
    }


@pytest.mark.parametrize("cfg, missing", [
    ({}, ["MNEMIQ_ORACLE_USER", "MNEMIQ_ORACLE_PASSWORD"]),
    ({"oracle_user": "app"}, ["MNEMIQ_ORACLE_PASSWORD"]),
    ({"oracle_password": "pw"}, ["MNEMIQ_ORACLE_USER"]),
])
def test_oracle_without_credentials_names_the_variables_it_needs(recorded, cfg, missing):
    """The message has to name the env vars. An Easy Connect descriptor has no credential slot,
    so an operator who configured one has no reason to expect a separate step exists."""
    with pytest.raises(resolve.SourceUnconfigured) as exc:
        resolve.adapter_for(_spec(kind="oracle", target="h:1521/PDB1"), _settings(**cfg))
    for name in missing:
        assert name in str(exc.value)


def test_oracle_is_refused_by_federation_with_a_message_that_explains_why():
    """DuckDB has no Oracle ATTACH scanner. This used to be `KeyError('oracle')` raised from an
    extension lookup, which names neither the source nor the reason."""
    from mnemiq.adapters.federated import FederatedAdapter, UnfederatableSource

    specs = [_spec(id="pg"), _spec(id="ora", kind="oracle", target="h:1521/PDB1")]
    with pytest.raises(UnfederatableSource) as exc:
        FederatedAdapter(specs)
    assert "'ora'" in str(exc.value) and "oracle" in str(exc.value)
    assert "postgres" in str(exc.value), "it should say what federation DOES support"


# -- refusals ----------------------------------------------------------------------------------

def test_an_unknown_kind_is_refused_by_name(recorded):
    with pytest.raises(resolve.UnknownSourceKind) as exc:
        resolve.adapter_for(_spec(kind="mysql"), _settings())
    assert "mysql" in str(exc.value)


def test_an_empty_target_is_refused_rather_than_attached(recorded):
    """`Settings.source_specs` synthesizes `target=self.pg_dsn or ""`, so an unconfigured engine
    produces a spec with an empty target rather than no spec at all."""
    with pytest.raises(resolve.SourceUnconfigured) as exc:
        resolve.adapter_for(_spec(target=""), _settings())
    # It must name the variable, not just the source: the default id `acme` was never chosen by
    # the operator and tells them nothing to do. `mnemiq enrich` on an unconfigured engine is the
    # first command anyone runs, and this is the message it prints.
    assert "MNEMIQ_PG_DSN" in str(exc.value)


def test_an_empty_target_for_a_kind_with_no_env_var_points_at_the_manifest(recorded):
    with pytest.raises(resolve.SourceUnconfigured) as exc:
        resolve.adapter_for(_spec(kind="oracle", target=""), _settings())
    assert "MNEMIQ_SOURCES_PATH" in str(exc.value) and "MNEMIQ_PG_DSN" not in str(exc.value)


# -- picking the source a single-source command acts on -----------------------------------------

def test_source_spec_picks_by_id_when_several_are_configured(tmp_path):
    manifest = tmp_path / "s.json"
    manifest.write_text(json.dumps([
        {"id": "a", "kind": "postgres", "target": "ta", "catalog": "a", "schema": "public"},
        {"id": "b", "kind": "postgres", "target": "tb", "catalog": "b", "schema": "public"},
    ]))
    s = _settings(sources_path=str(manifest), source_id="b")
    assert resolve.source_spec(s).target == "tb"


def test_source_spec_lists_the_ids_when_the_configured_one_matches_none(tmp_path):
    manifest = tmp_path / "s.json"
    manifest.write_text(json.dumps([
        {"id": "a", "kind": "postgres", "target": "ta", "catalog": "a", "schema": "public"},
        {"id": "b", "kind": "postgres", "target": "tb", "catalog": "b", "schema": "public"},
    ]))
    s = _settings(sources_path=str(manifest), source_id="ghost")
    with pytest.raises(resolve.SourceUnconfigured) as exc:
        resolve.source_spec(s)
    assert "ghost" in str(exc.value) and "a, b" in str(exc.value)


# -- the door the bug was actually behind --------------------------------------------------------

def test_build_runtime_dispatches_on_the_spec_not_on_pg_dsn(monkeypatch, tmp_path):
    """The resolver tests above exercise `adapter_for`. The BUG was one level up.

    `build_runtime`'s single-source branch read `settings.pg_dsn` directly, so it could have been
    corrected in the resolver and still connected to the wrong database. Only a test through this
    door distinguishes the two, and no non-live test built a runtime before this one.

    A sentinel exception is the observation point: it passes through untouched, where
    `SourceUnconfigured` would be translated to `SnapshotMissing`, and it avoids standing up the
    LLM client and the rest of the runtime just to read one argument.
    """
    from mnemiq.contract.semantic import Snapshot
    from mnemiq.runtime import build_runtime
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    store = tmp_path / "s.duckdb"
    con = init_store(str(store))
    save_snapshot(con, Snapshot(version="v1", source_id="only", created_at="t"))
    con.close()

    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "sqlite", "target": "/data/one.db",
                                     "catalog": "s", "schema": "main"}]))
    # source_id is deliberately NOT "only": the snapshot above is keyed by the manifest's id, and
    # pinning them equal here would hide whether the lookup reconciles them.
    s = _settings(sources_path=str(manifest), pg_dsn="postgresql://decoy/decoy",
                  store_path=str(store))

    class _Reached(Exception):
        pass

    seen = {}

    def _spy(spec, settings=None, *, read_only=True):
        seen.update(id=spec.id, kind=spec.kind, target=spec.target, read_only=read_only)
        raise _Reached

    monkeypatch.setattr("mnemiq.adapters.resolve.adapter_for", _spy)
    with pytest.raises(_Reached):
        build_runtime(s)

    assert seen == {"id": "only", "kind": "sqlite", "target": "/data/one.db", "read_only": True}


def test_a_manifest_typo_reaches_the_operator_as_a_sentence(monkeypatch, tmp_path):
    """`mnemiq ask` and `mnemiq write` catch SnapshotMissing and print it. An unknown `kind` is a
    config typo, so it has to arrive through that same handler rather than as a traceback."""
    from mnemiq.contract.semantic import Snapshot
    from mnemiq.runtime import SnapshotMissing, build_runtime
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    store = tmp_path / "s.duckdb"
    con = init_store(str(store))
    save_snapshot(con, Snapshot(version="v1", source_id="only", created_at="t"))
    con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "postgrez", "target": "t",
                                     "catalog": "s", "schema": "main"}]))
    s = _settings(sources_path=str(manifest), source_id="only", store_path=str(store))

    with pytest.raises(SnapshotMissing) as exc:
        build_runtime(s)
    assert "postgrez" in str(exc.value)


def test_a_manifest_id_survives_enrich_to_ask(tmp_path):
    """The snapshot `enrich` writes must be the one `build`, `refresh` and `ask` find.

    `enrich` keys it by the SPEC's id. Every single-source lookup keyed on `settings.source_id`,
    which defaults to "acme" and is never reconciled with a manifest's id -- so a one-entry
    manifest naming anything else produced a successful enrich followed by "no snapshot -- run
    `mnemiq enrich` first" from every other command, permanently. Every MULTI-source branch
    already keyed on spec.id, so the two halves of the same file disagreed.

    This asserts the reconciliation directly, at the function the runtime and the CLI both reach.
    """
    from mnemiq.contract.semantic import Snapshot
    from mnemiq.runtime import load_current_snapshot
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    con = init_store(str(tmp_path / "s.duckdb"))
    save_snapshot(con, Snapshot(version="v1", source_id="warehouse", created_at="t"))
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "warehouse", "kind": "duckdb", "target": "/w.duckdb",
                                     "catalog": "w", "schema": "main"}]))
    s = _settings(sources_path=str(manifest))
    assert s.source_id == "acme", "precondition: the default id does not match the manifest"

    snap, versions = load_current_snapshot(s, con)
    assert (snap.source_id, versions) == ("warehouse", {"warehouse": "v1"})


# -- the source-enforcement advisory -------------------------------------------------------------

def test_an_adapter_that_can_assess_enforcement_is_asked_at_boot(caplog):
    """`assert_enforcing` was written, tested across four verdicts, and called by NOTHING outside
    its own tests -- the **M26** shape, found by an adversarial review of the lane that built it.

    Warns rather than refuses, deliberately: under the 2026-08-29 direction the engine still
    enforces, so a `bypassing` connection is one where mnemiq's filters are in force and refusing
    to boot would take down a working deployment over a control that is not yet load-bearing.
    Fail-closed lands with delegation (M57).
    """
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _Bypassing:
        def assert_enforcing(self):
            return "bypassing", "this principal holds EXEMPT ACCESS POLICY"

    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(_Bypassing())
    assert "bypassing" in caplog.text and "EXEMPT ACCESS POLICY" in caplog.text


def test_both_advisories_are_reported_and_a_clean_read_only_basis_does_not_warn(caplog):
    """The seam carries two questions now. The second exists because the statement gate provably
    cannot see a write reached through an AUTONOMOUS_TRANSACTION function behind a view."""
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _Both:
        def assert_enforcing(self):
            return "attached", "every visible table carries an enabled SELECT policy"

        def assert_read_only(self):
            return "gate_only", "this read-only connection CAN write: it owns 3 table(s)"

    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(_Both())
    assert "gate_only" in caplog.text and "CAN write" in caplog.text
    assert "attached" not in caplog.text, "a clean enforcement verdict must not warn"


def test_a_source_that_is_enforcing_does_not_warn(caplog):
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _Attached:
        def assert_enforcing(self):
            return "attached", "every visible table carries an enabled SELECT policy"

    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(_Attached())
    assert caplog.text == "", "a clean verdict must not cry wolf at every boot"


def test_an_adapter_without_the_capability_is_not_an_error(caplog):
    """Three of the four adapters cannot answer this. Absence is not a failure -- the distinction
    this register keeps re-learning."""
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(object())
    assert caplog.text == ""


def test_a_source_that_will_not_answer_does_not_stop_the_engine_booting(caplog):
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _Broken:
        def assert_enforcing(self):
            raise RuntimeError("ORA-00942: table or view does not exist")

    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(_Broken())  # must not raise
    assert "could not assess" in caplog.text and "ORA-00942" in caplog.text
