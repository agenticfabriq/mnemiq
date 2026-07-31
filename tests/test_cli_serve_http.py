from mnemiq.cli import build_parser


def test_serve_parses_http_flags():
    args = build_parser().parse_args(["serve", "--http", "--host", "0.0.0.0", "--port", "9000"])
    assert args.http is True and args.host == "0.0.0.0" and args.port == 9000


def test_serve_defaults_to_stdio_mcp():
    args = build_parser().parse_args(["serve"])
    assert args.http is False


def test_serve_http_builds_runtime_then_runs_uvicorn(monkeypatch):
    import mnemiq.server.app as server_app

    calls = {}
    monkeypatch.setattr("mnemiq.runtime.build_runtime", lambda s: "RT")
    monkeypatch.setattr(server_app, "build_app", lambda rt, ident, **kw: ("APP", rt))
    monkeypatch.setattr(server_app, "_run_uvicorn",
                        lambda app, host, port: calls.update(app=app, host=host, port=port))
    server_app.serve_http(settings=None, host="127.0.0.1", port=8080)
    assert calls["app"][1] == "RT" and calls["port"] == 8080
