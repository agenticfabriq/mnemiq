from mnemiq.cli import build_parser, main
from mnemiq.config import Settings


def _fake_settings():
    from mnemiq.config import Settings

    return Settings(
        llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d", acme_data_dir=None
    )


def test_parser_exposes_the_lifecycle_subcommands():
    p = build_parser()
    args = p.parse_args(["ask", "how many claims?", "--json", "--mode", "deep"])
    assert args.command == "ask" and args.question == "how many claims?" and args.json is True
    assert args.mode == "deep"
    assert p.parse_args(["ask", "q"]).mode is None  # unset -> the router decides
    w = p.parse_args(["write", "INSERT INTO claim (id) VALUES (1)", "--json"])
    assert w.command == "write" and w.json is True
    for cmd in ("enrich", "build", "serve", "eval"):
        assert p.parse_args([cmd]).command == cmd  # each parses with no extra args


def test_ask_prints_answer_and_exits_zero(monkeypatch, capsys):
    import mnemiq.cli as cli
    from mnemiq.agent.loop import AgentAnswer

    class _RT:
        def ask(self, q, identity, mode=None):
            return AgentAnswer(answer="There are 2 claims.", trace=None, deferred=False)

    monkeypatch.setattr(cli, "build_runtime", lambda settings: _RT())
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: _fake_settings()))
    code = main(["ask", "how many claims?"])
    assert code == 0
    assert "There are 2 claims." in capsys.readouterr().out


def test_ask_on_a_deferral_still_exits_zero(monkeypatch, capsys):
    import mnemiq.cli as cli
    from mnemiq.agent.loop import AgentAnswer

    class _RT:
        def ask(self, q, identity, mode=None):
            return AgentAnswer(answer="I cannot answer that.", deferred=True)

    monkeypatch.setattr(cli, "build_runtime", lambda settings: _RT())
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: _fake_settings()))
    assert main(["ask", "salary?"]) == 0  # a deferral is a valid answer, not an error


def test_ask_reports_missing_snapshot_as_nonzero(monkeypatch, capsys):
    import mnemiq.cli as cli
    from mnemiq.runtime import SnapshotMissing

    def _boom(settings):
        raise SnapshotMissing("run `mnemiq enrich` then `mnemiq build` first")

    monkeypatch.setattr(cli, "build_runtime", _boom)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: _fake_settings()))
    assert main(["ask", "q"]) == 1
    assert "mnemiq enrich" in capsys.readouterr().err


def test_ask_rejects_an_unknown_mode_at_the_parser():
    import pytest

    with pytest.raises(SystemExit):  # argparse choices: fail closed before any work
        build_parser().parse_args(["ask", "q", "--mode", "fastest"])


def test_ask_passes_mode_to_the_runtime_and_reports_it(monkeypatch, capsys):
    import mnemiq.cli as cli
    from mnemiq.agent.loop import AgentAnswer

    class _RT:
        def ask(self, q, identity, mode=None):
            self.mode = mode
            return AgentAnswer(answer="ok", mode=mode or "thinking")

    rt = _RT()
    monkeypatch.setattr(cli, "build_runtime", lambda settings: rt)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: _fake_settings()))
    assert main(["ask", "q", "--mode", "instant", "--json"]) == 0
    assert rt.mode == "instant"
    assert '"mode": "instant"' in capsys.readouterr().out


def test_write_reports_a_refusal_and_exits_zero(monkeypatch, capsys):
    import mnemiq.cli as cli
    from mnemiq.runtime import WriteResult

    class _RT:
        def write(self, sql, identity):
            return WriteResult(approved=False, refusal="You may not write to 'claim'.")

    monkeypatch.setattr(cli, "build_runtime", lambda settings: _RT())
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: _fake_settings()))
    assert main(["write", "INSERT INTO claim (id) VALUES (1)"]) == 0  # a refusal is valid
    assert "may not write" in capsys.readouterr().out


def test_cmd_enrich_lazy_imports_resolve(capsys):
    # Regression (plan-29): _cmd_enrich's import block runs BEFORE the pg_dsn check, so a
    # no-DSN Settings still exercises every lazy import. Folding bird_runner._flag into
    # settings.enrich_facts/examples broke `import _flag` here until enrich was updated.
    from mnemiq.cli import _cmd_enrich
    from mnemiq.config import Settings
    assert _cmd_enrich(Settings()) == 1  # no pg_dsn -> 1, but only after imports resolve
    assert "MNEMIQ_PG_DSN" in capsys.readouterr().err


def test_digest_ontology_writes_records(tmp_path):
    import json

    from mnemiq.cli import main

    out = tmp_path / "records.json"
    rc = main(["digest-ontology", "--ttl", "tests/fixtures/ontology/skos_scheme.ttl",
               "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["version"]
    assert any(s["label"] == "Colour Codes" for s in payload["schemes"])


# ---------------------------------------------------------------------------
# Issue #4: MNEMIQ_ROLES (and MNEMIQ_PRINCIPAL) must be honoured by the CLI
# when --roles / --principal is not explicitly passed.
# ---------------------------------------------------------------------------


def test_roles_default_from_env_when_flag_omitted(monkeypatch, capsys):
    """The defect: MNEMIQ_ROLES=analyst was ignored because --roles defaulted to empty,
    producing no grants. The identity should carry the env var's roles when the flag is absent."""
    import mnemiq.cli as cli
    from mnemiq.agent.loop import AgentAnswer

    captured = {}

    class _RT:
        def ask(self, q, identity, mode=None):
            captured["identity"] = identity
            return AgentAnswer(answer="ok", trace=None, deferred=False)

    settings = Settings(
        llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d",
        roles="analyst,viewer",
    )
    monkeypatch.setattr(cli, "build_runtime", lambda s: _RT())
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    assert main(["ask", "q"]) == 0
    assert captured["identity"].roles == ["analyst", "viewer"]


def test_roles_flag_overrides_env_var(monkeypatch, capsys):
    """An explicit --roles flag must override MNEMIQ_ROLES, not merge with it."""
    import mnemiq.cli as cli
    from mnemiq.agent.loop import AgentAnswer

    captured = {}

    class _RT:
        def ask(self, q, identity, mode=None):
            captured["identity"] = identity
            return AgentAnswer(answer="ok", trace=None, deferred=False)

    settings = Settings(
        llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d",
        roles="analyst",
    )
    monkeypatch.setattr(cli, "build_runtime", lambda s: _RT())
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    assert main(["ask", "q", "--roles", "admin"]) == 0
    assert captured["identity"].roles == ["admin"]


def test_principal_defaults_from_env_when_flag_omitted(monkeypatch, capsys):
    """Same fallback for --principal: MNEMIQ_PRINCIPAL should be honoured."""
    import mnemiq.cli as cli
    from mnemiq.agent.loop import AgentAnswer

    captured = {}

    class _RT:
        def ask(self, q, identity, mode=None):
            captured["identity"] = identity
            return AgentAnswer(answer="ok", trace=None, deferred=False)

    settings = Settings(
        llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d",
        principal="alice@corp.com",
    )
    monkeypatch.setattr(cli, "build_runtime", lambda s: _RT())
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    assert main(["ask", "q"]) == 0
    assert captured["identity"].principal_id == "alice@corp.com"


def test_write_honours_roles_from_env(monkeypatch, capsys):
    """The write path had the same gap: --roles defaulted to empty, ignoring MNEMIQ_ROLES."""
    import mnemiq.cli as cli
    from mnemiq.runtime import WriteResult

    captured = {}

    class _RT:
        def write(self, sql, identity):
            captured["identity"] = identity
            return WriteResult(approved=False, refusal="denied")

    settings = Settings(
        llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d",
        roles="writer",
    )
    monkeypatch.setattr(cli, "build_runtime", lambda s: _RT())
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    assert main(["write", "INSERT INTO t (id) VALUES (1)"]) == 0
    assert captured["identity"].roles == ["writer"]


def test_tenant_defaults_from_env(monkeypatch, capsys):
    """MNEMIQ_TENANT was hardcoded to 'local' in _identity. Now that we delegate to
    identity_from_settings, tenant is honoured the same way as principal and roles."""
    import mnemiq.cli as cli
    from mnemiq.agent.loop import AgentAnswer

    captured = {}

    class _RT:
        def ask(self, q, identity, mode=None):
            captured["identity"] = identity
            return AgentAnswer(answer="ok", trace=None, deferred=False)

    settings = Settings(
        llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d",
        tenant="acme",
    )
    monkeypatch.setattr(cli, "build_runtime", lambda s: _RT())
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    assert main(["ask", "q"]) == 0
    assert captured["identity"].tenant_id == "acme"


def test_roles_flag_strips_names_and_an_empty_flag_still_means_no_roles(monkeypatch):
    """The `--roles` path strips like the environment path, and the two flags keep falling back by
    different rules on purpose.

    `--roles ""` is a real instruction -- grant nothing -- so it beats a configured role set; an
    empty `--principal` is not an instruction, so it falls through rather than building an identity
    with no principal.

    Each assertion holds a DIFFERENT property, measured by mutating `_identity` one way at a time
    rather than reasoned about -- an earlier version of this paragraph named its control by
    position, a later commit inserted a case at that position, and the correction then re-pointed
    it at two assertions that guard something else:

      * `--roles ""` -> `[]` is the ONLY guard that an explicit empty flag beats a configured role
        set. Making an empty parse fall through (`parsed or base.roles`) fails this line alone.
      * bare `ask q` -> `["analyst"]` guards the opposite direction, that an ABSENT flag falls
        through to settings. Returning `[]` there fails this and the env-fallback tests above.
      * `--roles "a, b"` guards that the flag is consulted at all; ignoring it entirely fails here
        first.
    """
    import mnemiq.cli as cli

    settings = Settings(
        llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d",
        roles="analyst", principal="alice@corp.com",
    )

    def ident(argv):
        return cli._identity(cli.build_parser().parse_args(argv), settings)

    assert ident(["ask", "q", "--roles", "a, b"]).roles == ["a", "b"]
    # An element that is nothing but whitespace is not a role. Without this the filter could go
    # back to `if r` and stay green, letting `--roles "a, ,b"` carry a "" role.
    assert ident(["ask", "q", "--roles", "a, ,b"]).roles == ["a", "b"]
    assert ident(["ask", "q", "--roles", ""]).roles == []
    assert ident(["ask", "q", "--principal", ""]).principal_id == "alice@corp.com"
    # The case the principal strip actually exists for. `--principal ""` above is resolved by the
    # `or` alone, so it reads the strip not at all -- deleting `.strip()` left every test green.
    assert ident(["ask", "q", "--principal", "   "]).principal_id == "alice@corp.com"
    # Controls: the flag still wins when given, and absence still falls through to settings.
    assert ident(["ask", "q", "--roles", "admin"]).roles == ["admin"]
    assert ident(["ask", "q"]).roles == ["analyst"]
