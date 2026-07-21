from mnemiq.cli import build_parser, main


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
