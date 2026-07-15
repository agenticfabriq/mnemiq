from mnemiq.cli import build_parser, main


def _fake_settings():
    from mnemiq.config import Settings

    return Settings(
        llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d", acme_data_dir=None
    )


def test_parser_exposes_the_lifecycle_subcommands():
    p = build_parser()
    args = p.parse_args(["ask", "how many claims?", "--json"])
    assert args.command == "ask" and args.question == "how many claims?" and args.json is True
    for cmd in ("enrich", "build", "serve", "eval"):
        assert p.parse_args([cmd]).command == cmd  # each parses with no extra args


def test_ask_prints_answer_and_exits_zero(monkeypatch, capsys):
    import mnemiq.cli as cli
    from mnemiq.agent.loop import AgentAnswer

    class _RT:
        def ask(self, q, identity):
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
        def ask(self, q, identity):
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
