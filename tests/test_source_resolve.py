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
        # None unless configured: the plain TCP path must not acquire TLS arguments it never had
        "config_dir": None, "wallet_password": None,
    }


def test_oracle_carries_a_config_dir_and_wallet_password_when_they_are_set(recorded):
    """One directory serves on-prem TNS aliases and Autonomous mTLS, so it reaches the adapter
    from settings the same way the credentials do."""
    s = _settings(oracle_user="app", oracle_password="pw",
                  oracle_config_dir="/etc/oracle/wallet", oracle_wallet_password="wp")
    kwargs = resolve.adapter_for(_spec(kind="oracle", target="alias"), s).kwargs
    assert kwargs["config_dir"] == "/etc/oracle/wallet"
    assert kwargs["wallet_password"] == "wp"


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


def test_a_minimal_read_principal_is_still_warned_that_read_only_is_not_a_guarantee(caplog):
    """**This test previously asserted the opposite, and that is the point of keeping it here.**

    It required a correctly configured deployment to boot SILENTLY, on the premise that
    `unverifiable` means "nothing further exists to do". The premise is false: default logging is
    WARNING, so the operator of exactly the deployment we recommend was told nothing, while a
    principal holding SELECT on one view can still cause a write.

    The action is not "narrow the principal" -- they already have -- it is to stop treating
    `read_only=True` as a guarantee and enforce read-only in the DATABASE, which is what the
    2026-08-29 direction says. A verdict that reports an unclosable gap has to reach the operator;
    the earlier noise complaint is answered by making the warning true and actionable, not by
    silencing it.
    """
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _Minimal:
        def assert_enforcing(self):
            return "attached", "every visible table carries an enabled SELECT policy"

        def assert_read_only(self):
            return "unverifiable", ("no write-shaped privilege was found, and that is NOT the same "
                                    "as being unable to write. Do not treat read_only=True as a "
                                    "guarantee here; enforce read-only in the DATABASE")

    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(_Minimal())
    assert "read-only basis" in caplog.text, "an unclosable gap must not be silent at the default level"
    assert "DATABASE" in caplog.text, "and it must name the action, or it is the noise it was called"
    assert "source enforcement" not in caplog.text, "a clean enforcement verdict still stays quiet"


def test_writable_is_the_only_quiet_read_only_verdict(caplog):
    """`writable` means the question does not apply -- the engine was asked to attach read-write.

    Every other verdict reports something the operator should know, so this is the one quiet case.
    """
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _Writable:
        def assert_read_only(self):
            return "writable", "this adapter is not read-only, so the question does not apply"

    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(_Writable())
    assert caplog.text == ""


def test_an_acknowledged_verdict_stops_warning_but_a_worse_one_still_does(caplog):
    """An unclosable gap must reach the operator once; it must not page them forever.

    Both failure modes have now happened here in consecutive commits -- warning on every boot of
    every correct deployment, then silencing it and hiding the gap entirely at the default level.
    Acknowledgement resolves them: the operator who has assessed the state records it and stops
    hearing about it.

    **Keyed on `<advisory>:<verdict>`, not on the advisory.** Acknowledging the advisory as a whole
    would silence a WORSE verdict arriving later -- `read-only basis` moving from `unverifiable`,
    the unclosable ceiling, to `gate_only`, meaning a principal that now holds write privilege. That
    transition is the one worth paging on, so it survives the acknowledgement that covers the state
    before it.
    """
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _Read:
        def __init__(self, verdict):
            self.verdict = verdict

        def assert_read_only(self):
            return self.verdict, "detail"

    ack = frozenset({"read-only-basis:unverifiable"})

    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(_Read("unverifiable"))
    assert "read-only basis" in caplog.text, "unacknowledged, the gap must be loud"

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(_Read("unverifiable"), ack)
    assert caplog.text == "", "acknowledged, that verdict is quiet"

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_source_enforcement(_Read("gate_only"), ack)
    assert "gate_only" in caplog.text, (
        "a DIFFERENT, worse verdict must still warn -- acknowledging a state must not "
        "acknowledge every future state of the same advisory"
    )


def test_acknowledgements_are_parsed_case_and_space_insensitively():
    from mnemiq.runtime import _acknowledged

    s = _settings(ack_advisories=" Read-Only-Basis:Unverifiable , source-enforcement:partial ")
    assert _acknowledged(s) == {"read-only-basis:unverifiable", "source-enforcement:partial"}
    assert _acknowledged(_settings()) == frozenset()
    assert _acknowledged(_settings(ack_advisories="  ,, ")) == frozenset()


def test_a_misspelled_acknowledgement_says_so_instead_of_doing_nothing_quietly():
    """A typo'd ack silences nothing and looks exactly like no ack -- two causes, one observable.

    That is the collapse this codebase keeps closing, and it appeared inside the mechanism added to
    answer a review about the advisory itself. The two halves are reported differently on purpose:
    an unknown ADVISORY name cannot be right, so it warns; an acknowledged VERDICT that did not
    occur is the normal case for a deployment whose state improved, so it is INFO. Warning on the
    second would rebuild the noise this setting exists to remove.
    """
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _Read:
        def assert_read_only(self):
            return "unverifiable", "detail"

    def run(ack, level):
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        lg = logging.getLogger("mnemiq.runtime")
        lg.addHandler(handler)
        previous, lg.level = lg.level, level
        try:
            _warn_source_enforcement(_Read(), frozenset(ack))
        finally:
            lg.removeHandler(handler)
            lg.level = previous
        return stream.getvalue()

    unknown = run({"read-only-baiss:unverifiable"}, logging.WARNING)
    assert "names no advisory" in unknown, "an advisory name that cannot be right must warn"
    assert "read-only basis: unverifiable" in unknown, "and the real warning still fires"

    typo = run({"read-only-basis:unverifialbe"}, logging.INFO)
    assert "did not apply this boot" in typo
    assert "read-only basis: unverifiable" in typo, "the gap is still reported"

    good = run({"read-only-basis:unverifiable"}, logging.INFO)
    assert "acknowledged via MNEMIQ_ACK_ADVISORIES" in good
    assert "names no advisory" not in good and "did not apply" not in good

    unused = run({"read-only-basis:gate_only"}, logging.INFO)
    assert "did not apply this boot" in unused, (
        "acknowledging a verdict that did not occur is legitimate -- the state may have improved -- "
        "so it is reported, not warned about"
    )


def test_an_ack_carried_to_an_adapter_that_cannot_answer_says_which_cause():
    """The third reason an acknowledgement goes unused, and the one that misleads.

    Only `OracleAdapter` implements `assert_read_only`/`assert_enforcing`. An operator who carries
    `MNEMIQ_ACK_ADVISORIES` from an Oracle deployment to a Postgres one is neither stale nor
    misspelled -- the check never ran -- and telling them "the verdict changed" sends them to look
    at a verdict that was never produced.
    """
    import io
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _CannotAnswer:
        """Every adapter but Oracle."""

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    lg = logging.getLogger("mnemiq.runtime")
    lg.addHandler(handler)
    previous, lg.level = lg.level, logging.INFO
    try:
        _warn_source_enforcement(_CannotAnswer(), frozenset({"read-only-basis:unverifiable"}))
    finally:
        lg.removeHandler(handler)
        lg.level = previous
    out = stream.getvalue()
    assert "produced no verdict" in out
    assert "not implemented here" in out, "the message must cover both ways no verdict appears"
    assert "the verdict changed" not in out, "that diagnosis would send them to the wrong place"


def test_an_advisory_that_raised_is_not_diagnosed_as_a_changed_verdict():
    """`assessed` was recorded BEFORE the call, so an advisory that raised counted as assessed.

    An unmatched acknowledgement for it was then told "the verdict changed, or the verdict half is
    misspelled" while the real cause -- the exception -- was logged two lines above. The wrong
    diagnosis and its correction sat in the same output.
    """
    import io
    import logging

    from mnemiq.runtime import _warn_source_enforcement

    class _Raises:
        def assert_read_only(self):
            raise RuntimeError("ORA-00942: table or view does not exist")

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    lg = logging.getLogger("mnemiq.runtime")
    lg.addHandler(handler)
    previous, lg.level = lg.level, logging.INFO
    try:
        _warn_source_enforcement(_Raises(), frozenset({"read-only-basis:unverifiable"}))
    finally:
        lg.removeHandler(handler)
        lg.level = previous
    out = stream.getvalue()
    assert "could not assess" in out and "ORA-00942" in out
    assert "produced no verdict" in out
    assert "the verdict changed" not in out


# -- what actually reaches the driver ------------------------------------------------------------
#
# These live HERE, not in tests/test_oracle_adapter.py, and that placement is the point. They need
# no database -- they mock `oracledb` outright -- but that module is gated on
# MNEMIQ_ORACLE_TEST_DSN and skips wholesale without a live instance, so the sole regression guard
# for "a plain connection acquires no TLS arguments" was inert in every ordinary CI run. A review
# caught it by collecting the file and seeing `0 items / 1 skipped`.


class _RecordingOracledb:
    """Stands in for the driver so the CONNECT ARGUMENTS can be asserted."""

    DatabaseError = Exception
    seen: dict = {}

    @classmethod
    def connect(cls, **kwargs):
        cls.seen = dict(kwargs)
        return _RecordingConnection()


class _RecordingConnection:
    def cursor(self):
        raise AssertionError("no statement should run during construction")


@pytest.fixture
def driver(monkeypatch):
    import sys

    _RecordingOracledb.seen = {}
    monkeypatch.setitem(sys.modules, "oracledb", _RecordingOracledb)
    return _RecordingOracledb


def test_a_plain_connection_passes_no_tls_arguments(driver):
    """Asserted on the CALL, not on a successful connection: a connection that works says nothing
    about which keywords reached the driver, and `oracledb.connect` treats an explicit
    `config_dir=None` differently from an absent one in some releases."""
    from mnemiq.adapters.oracle import OracleAdapter

    OracleAdapter(dsn="h:1521/S", user="u", password="p")
    assert set(driver.seen) == {"user", "password", "dsn"}, f"unexpected: {sorted(driver.seen)}"


def test_a_configured_directory_reaches_the_driver_as_both_config_and_wallet_location(driver):
    from mnemiq.adapters.oracle import OracleAdapter

    OracleAdapter(dsn="alias", user="u", password="p", config_dir="/w", wallet_password="wp")
    assert driver.seen["config_dir"] == "/w"
    assert driver.seen["wallet_location"] == "/w", "thin mode reads the PEM from the wallet location"
    assert driver.seen["wallet_password"] == "wp"


def test_a_wallet_password_without_a_directory_is_refused_by_name(driver):
    """Sending a wallet password with nowhere to find a wallet is a misconfiguration, and it was
    passed to the driver silently. Named the way the missing-credential path names its variables,
    because the operator's next move is to set one."""
    from mnemiq.adapters.oracle import OracleAdapter

    with pytest.raises(ValueError) as exc:
        OracleAdapter(dsn="h:1521/S", user="u", password="p", wallet_password="wp")
    assert "MNEMIQ_ORACLE_CONFIG_DIR" in str(exc.value)
    assert "MNEMIQ_ORACLE_WALLET_PASSWORD" in str(exc.value)
    assert driver.seen == {}, "it must refuse before opening a connection"


def test_a_half_configured_wallet_arrives_as_a_clean_message_not_a_traceback(driver):
    """`ask`, `write`, `enrich` and `refresh` all catch `SourceUnconfigured` and print it.

    The adapter validates its own TLS arguments and raises `ValueError`, which none of those doors
    catch -- so the operator got a traceback for a plain misconfiguration, while the sibling
    missing-credential case one function up produced a sentence. It is translated at the resolver
    because the adapter cannot import this module: the dependency runs one way.
    """
    s = _settings(oracle_user="app", oracle_password="pw", oracle_wallet_password="wp")
    with pytest.raises(resolve.SourceUnconfigured) as exc:
        resolve.adapter_for(_spec(kind="oracle", target="h:1521/S"), s)
    assert "MNEMIQ_ORACLE_CONFIG_DIR" in str(exc.value)


def test_build_runtime_renders_it_the_same_way_as_any_other_misconfiguration(monkeypatch, tmp_path):
    """The door, not just the resolver -- `build_runtime` translates `SourceUnconfigured` into
    `SnapshotMissing`, which is what the CLI actually prints."""
    from mnemiq.contract.semantic import Snapshot
    from mnemiq.runtime import SnapshotMissing, build_runtime
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    store = tmp_path / "s.duckdb"
    con = init_store(str(store))
    save_snapshot(con, Snapshot(version="v1", source_id="only", created_at="t"))
    con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "oracle", "target": "h:1521/S",
                                     "catalog": "o", "schema": "APP"}]))
    s = _settings(sources_path=str(manifest), source_id="only", store_path=str(store),
                  oracle_user="app", oracle_password="pw", oracle_wallet_password="wp")
    with pytest.raises(SnapshotMissing, match="MNEMIQ_ORACLE_CONFIG_DIR"):
        build_runtime(s)


def test_traces_and_metrics_are_attributed_to_the_manifest_source_not_the_setting():
    """The resolver's claim is that a manifest id survives end to end. It survived as far as the
    snapshot and stopped: `load_current_snapshot` keys on `specs[0].id`, while the answer trace and
    the metrics record both still used `settings.source_id` -- so a one-entry manifest naming
    `warehouse` answered over `warehouse` and recorded every answer under `acme`.

    Attribution is not a cosmetic label. A trace is evidence about WHICH source answered.
    """
    from mnemiq.contract.semantic import Snapshot
    from mnemiq.runtime import Runtime

    rt = object.__new__(Runtime)
    rt.snapshot = Snapshot(version="v1", source_id="warehouse", created_at="t")
    rt.settings = _settings(source_id="acme")
    assert rt._source_id() == "warehouse"

    # and it degrades the way the old code did when there is no snapshot to ask
    rt.snapshot = None
    assert rt._source_id() == "acme"
    rt.settings = None
    assert rt._source_id() == "unknown"


def _metrics_asks_for(monkeypatch, tmp_path, capsys, specs, snapshot_ids, source_id="acme"):
    """Run `mnemiq metrics` against a store holding one snapshot per id, and report what it queried.

    Federation needs a snapshot PER SOURCE -- `load_current_snapshot` resolves a version for each
    spec and merges them, and the merged result is what carries `federated`. Saving a single
    pre-merged snapshot is not the shape the runtime ever sees.
    """
    from mnemiq.cli import _cmd_metrics
    from mnemiq.contract.semantic import Snapshot
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    store = tmp_path / "s.duckdb"
    con = init_store(str(store))
    for i, sid in enumerate(snapshot_ids):
        save_snapshot(con, Snapshot(version=f"v{i}", source_id=sid, created_at="t"))
    con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps(specs))

    asked: list = []

    class _Sink:
        def recent(self, sid, n):
            asked.append(sid)
            return []

    monkeypatch.setattr("mnemiq.observability.metrics.NullSink", lambda: _Sink())
    _cmd_metrics(_settings(sources_path=str(manifest), source_id=source_id,
                           store_path=str(store)))
    capsys.readouterr()
    return asked


def test_metrics_reads_the_id_a_single_source_manifest_is_recorded_under(
    monkeypatch, tmp_path, capsys
):
    """`Runtime._source_id` records under the SNAPSHOT's id. Reading `settings.source_id` meant
    answers written under `warehouse` were queried as `acme` -- an EMPTY view, not a wrong one."""
    asked = _metrics_asks_for(
        monkeypatch, tmp_path, capsys,
        [{"id": "warehouse", "kind": "duckdb", "target": "/w.duckdb", "catalog": "w",
          "schema": "main"}],
        snapshot_ids=["warehouse"])
    assert asked == ["warehouse"]


def test_metrics_reads_the_federated_id_and_not_a_member_of_the_federation(
    monkeypatch, tmp_path, capsys
):
    """The second wrong answer, and why this now derives the id from `load_current_snapshot`
    instead of restating the rule.

    A merged snapshot is recorded under `federated`. `source_spec(settings).id` -- my first fix --
    answers a DIFFERENT question, which single source a one-source command acts on, and returned a
    member id like `pg`. Both wrong answers produced an empty view rather than a wrong one.
    """
    asked = _metrics_asks_for(
        monkeypatch, tmp_path, capsys,
        [{"id": "pg", "kind": "postgres", "target": "t", "catalog": "a", "schema": "public"},
         {"id": "lite", "kind": "sqlite", "target": "t", "catalog": "b", "schema": "main"}],
        snapshot_ids=["pg", "lite"], source_id="pg")
    assert asked == ["federated"], f"asked {asked}, but the runtime records under 'federated'"


def test_metrics_still_prints_when_there_is_no_store_to_ask(monkeypatch, tmp_path, capsys):
    """A metrics view must not fail to print because nothing has been enriched yet."""
    from mnemiq.cli import _cmd_metrics

    asked: list = []

    class _Sink:
        def recent(self, sid, n):
            asked.append(sid)
            return []

    monkeypatch.setattr("mnemiq.observability.metrics.NullSink", lambda: _Sink())
    assert _cmd_metrics(_settings(pg_dsn="postgresql://h/db", source_id="acme",
                                  store_path=str(tmp_path / "absent.duckdb"))) == 0
    capsys.readouterr()
    assert asked == ["acme"]
