import pytest

from mnemiq.config import Settings


def _s(**kw) -> Settings:
    return Settings(local_only=True, **kw)


def test_a_loopback_endpoint_is_allowed():
    _s(llm_base_url="http://127.0.0.1:8000/v1", llm_api_key="k").assert_local_only()
    _s(llm_base_url="http://localhost:8000/v1", llm_api_key="k").assert_local_only()


def test_a_private_range_endpoint_is_allowed():
    # A DC deployment puts vLLM on another host on the same private network.
    _s(llm_base_url="http://10.4.1.9:8000/v1", llm_api_key="k").assert_local_only()
    _s(llm_base_url="http://192.168.8.20:8000/v1", llm_api_key="k").assert_local_only()


def test_a_public_endpoint_is_refused_and_named():
    s = _s(llm_base_url="https://api.openai.com/v1", llm_api_key="k")
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    assert "llm_base_url" in str(exc.value)
    assert "api.openai.com" in str(exc.value)


def test_every_offender_is_named_not_just_the_first():
    # An operator who fixes one and re-runs, only to be told about the next, will assume the
    # check is flaky and disable it.
    s = _s(
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_api_key="k",
        embed_base_url="https://api.openai.com/v1",
        verify_base_url="https://example.com/v1",
    )
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    assert "embed_base_url" in str(exc.value)
    assert "verify_base_url" in str(exc.value)


def test_the_check_is_off_unless_asked_for():
    Settings(llm_base_url="https://api.openai.com/v1", llm_api_key="k").assert_local_only()


def test_an_unparseable_url_is_refused_rather_than_allowed():
    # Fail closed: a URL we cannot classify is not evidence that it is local.
    with pytest.raises(RuntimeError):
        _s(llm_base_url="not a url", llm_api_key="k").assert_local_only()


def test_a_cloud_metadata_style_link_local_address_is_refused():
    # 169.254.169.254 is the well-known address several cloud providers use to serve their
    # metadata API. Python's ipaddress.is_private says True for the whole 169.254.0.0/16 block,
    # which would wave this through as "local" -- exactly the off-box hop this check exists to
    # catch. The docstring promises RFC1918 + loopback specifically, not `is_private`'s broader
    # IANA special-purpose set, and this is the case that distinguishes the two.
    with pytest.raises(RuntimeError) as exc:
        _s(llm_base_url="http://169.254.169.254/v1", llm_api_key="k").assert_local_only()
    assert "169.254.169.254" in str(exc.value)


def test_the_third_rfc1918_block_is_allowed():
    # The other two tested ranges are 10/8 and 192.168/16; 172.16/12 is the one a narrowed,
    # explicit RFC1918 check (rather than delegating to is_private) is most likely to drop.
    _s(llm_base_url="http://172.16.5.1:8000/v1", llm_api_key="k").assert_local_only()


def test_ipv6_loopback_is_allowed():
    _s(llm_base_url="http://[::1]:8000/v1", llm_api_key="k").assert_local_only()


def test_the_cli_calls_the_assertion_before_any_command_does_work(monkeypatch, capsys):
    # `--help` (the brief's original sketch) exits inside argparse, before `settings` is even
    # resolved -- it would pass whether or not cli.main calls the assertion at all. This drives
    # a REAL command instead (`metrics`, the cheapest one: no positional args, no source/LLM
    # config needed to reach its first line of work) and proves ORDER, not just that the method
    # ran: assert_local_only raises before returning, so if the call site in cli.main truly sits
    # immediately after `settings = Settings.from_env()` -- before the command dispatch -- nothing
    # past it can execute. `init_store` is patched to raise its own, differently-typed error as a
    # tripwire for a wiring regression that lets real work start. That tripwire does NOT surface as
    # an uncaught exception, though: `_cmd_metrics` wraps its own init_store call in a broad
    # `except Exception`, which catches the AssertionError, prints a warning and returns 0 -- so a
    # regression here is caught by `rc == 2` and `call_order` below, not by pytest.raises. Verified
    # by mutation: commenting out the real call site makes this test fail with `rc == 0` and
    # `call_order == ["init_store"]`, never with an uncaught error.
    call_order: list[str] = []

    def fake_assert_local_only(self) -> None:
        call_order.append("assert_local_only")
        raise RuntimeError("stopped-before-work")

    def fake_init_store(*args, **kwargs):
        call_order.append("init_store")
        raise AssertionError("real work started before assert_local_only ran")

    monkeypatch.setattr(Settings, "assert_local_only", fake_assert_local_only)
    monkeypatch.setattr("mnemiq.store.bootstrap.init_store", fake_init_store)

    from mnemiq import cli

    rc = cli.main(["metrics"])
    assert rc == 2, "a refused startup is a clean exit code, not a raised exception out of main"
    assert call_order == ["assert_local_only"]
    assert "stopped-before-work" in capsys.readouterr().err
