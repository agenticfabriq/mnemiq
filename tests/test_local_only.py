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


def test_the_check_is_off_unless_asked_for(monkeypatch):
    # Fix round 1, item 6a: `Settings(...)` without an explicit `local_only=` still reads
    # MNEMIQ_LOCAL_ONLY from the real environment (pydantic-settings, not just os.environ at
    # `from_env()` time) -- so `MNEMIQ_LOCAL_ONLY=1 pytest tests/test_local_only.py` made this
    # test fail on an unrelated public URL. Clearing the var is what actually asserts "off by
    # default" instead of "off in whatever environment happened to run this".
    monkeypatch.delenv("MNEMIQ_LOCAL_ONLY", raising=False)
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


# --- fix round 2, item 3: IPv6 unique-local addresses (RFC 4193, RFC1918's IPv6 analog) -----


def test_an_ipv6_unique_local_address_is_allowed():
    # fd00::/8 and fc00::/7 are RFC 4193's "unique local address" range -- the IPv6 equivalent
    # of RFC1918. `in_rfc1918` used to be gated on `addr.version == 4`, so an IPv6-only data
    # centre's own private host was refused AND told "is publicly routable", which is false:
    # `ipaddress.ip_address("fd00::1").is_global` is False.
    _s(llm_base_url="http://[fd00::1]:8000/v1", llm_api_key="k").assert_local_only()
    _s(llm_base_url="http://[fc00::1234]:8000/v1", llm_api_key="k").assert_local_only()


def test_a_genuinely_public_ipv6_address_is_still_called_publicly_routable():
    # This does NOT exercise the ULA fix itself -- fd00::/8 and fc00::/7 are never refused now,
    # so there is no refused-ULA case left to check the wording against. It only pins that a
    # clearly public address keeps earning that label -- it does NOT, by itself, prove `in_ula`
    # is exactly fc00::/7 and no wider: several ways to widen it (say to `addr.version == 6`, or
    # to fc00::/6) would still refuse this specific address and leave this test green. The
    # adjacent test below is the one that actually pins the boundary.
    s = _s(llm_base_url="http://[2001:4860:4860::8888]:8000/v1", llm_api_key="k")
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    assert "publicly routable" in str(exc.value)


def test_the_address_immediately_above_the_ula_range_is_still_refused():
    # fe00:: is the first address past fc00::/7 (which ends at fdff:ffff:...:ffff) -- the
    # tightest available probe against the UPPER edge of `in_ula` being accidentally widened,
    # e.g. to fc00::/6 (which WOULD wrongly include fe00::). Pins the upper boundary only; see
    # the sibling test below for the lower one. Neither the "ULA is allowed" tests above nor the
    # genuinely-public test above them would catch a /6 widening; this one does.
    s = _s(llm_base_url="http://[fe00::1]:8000/v1", llm_api_key="k")
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    assert "publicly routable" in str(exc.value)


def test_the_address_immediately_below_the_ula_range_is_still_refused():
    # fbff:ffff:...:ffff is the last address before fc00::/7 starts -- the tightest available
    # probe against the LOWER edge, e.g. a hand-widened range starting earlier than fc00::. No
    # other test in this file would catch that: the ULA-allowed tests use fd00::1/fc00::1234, the
    # genuinely-public test uses a routable address far outside either edge, and the sibling test
    # above only pins the upper edge.
    s = _s(llm_base_url="http://[fbff:ffff:ffff:ffff:ffff:ffff:ffff:ffff]:8000/v1", llm_api_key="k")
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    assert "publicly routable" in str(exc.value)


def test_an_ipv4_mapped_ipv6_private_address_is_allowed():
    # `::ffff:10.0.0.1` is RFC1918's 10.0.0.1 spelled as an IPv4-mapped IPv6 literal -- the kind
    # of address an IPv6-preferring resolver or proxy can hand back for an ordinary private host.
    # `ipaddress.ip_address("::ffff:10.0.0.1").is_private` is True and `.version` is 6, so the
    # old `addr.version == 4` gate refused it exactly like a ULA.
    _s(llm_base_url="http://[::ffff:10.0.0.1]:8000/v1", llm_api_key="k").assert_local_only()


def test_an_ipv4_mapped_ipv6_public_address_is_still_refused():
    s = _s(llm_base_url="http://[::ffff:8.8.8.8]:8000/v1", llm_api_key="k")
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    assert "llm_base_url" in str(exc.value)


# --- fix round 1, item 1: `urlparse` itself can raise -------------------------------------
#
# `urlparse("http://[::1/v1")` (a malformed IPv6 host literal) raises ValueError from INSIDE
# urlparse, before `.hostname` is ever touched -- the old code only wrapped the later
# `ipaddress.ip_address(host)` call in try/except, so this exception was uncaught. It then did
# two things wrong at once: it left `assert_local_only` as a raw traceback instead of the clean
# RuntimeError the rest of this module promises, and -- worse -- it discarded every offender the
# loop had already collected for an EARLIER field, because the exception unwound straight out of
# the function before the final `if offenders: raise` ran.
_UNPARSEABLE_URLS = [
    "http://[::1/v1",                  # unterminated IPv6 bracket
    "http://user:pass@[10.0.0.1]/v1",  # an IPv4 literal inside IPv6 brackets
    "http://10.0.0.1]/v1",             # a stray close-bracket with none to open it
    "http://a[b/v1",                   # a bare bracket in the host
    "http://℀.com/v1",            # NFKC-hostile codepoint (U+2100 ACCOUNT OF)
]


@pytest.mark.parametrize("bad_url", _UNPARSEABLE_URLS)
def test_a_url_urlparse_itself_rejects_is_refused_not_raised_uncaught(bad_url):
    with pytest.raises(RuntimeError) as exc:
        _s(llm_base_url=bad_url, llm_api_key="k").assert_local_only()
    assert "llm_base_url" in str(exc.value)


def test_an_unparseable_url_does_not_discard_an_earlier_offender():
    # The concrete failure the review reported: MNEMIQ_LLM_BASE_URL pointed at a real public
    # endpoint, MNEMIQ_EMBED_BASE_URL had a typo'd IPv6 literal, and the operator was told only
    # about the crash -- never about the OpenAI endpoint that was ALSO wrong.
    s = _s(
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="k",
        embed_base_url="http://[::1/v1",
    )
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    msg = str(exc.value)
    assert "llm_base_url" in msg and "api.openai.com" in msg
    assert "embed_base_url" in msg


# --- fix round 1, item 2: a passing check must say so ---------------------------------------


def test_a_passing_check_names_the_endpoints_it_cleared(monkeypatch, capsys):
    # MNEMIQ_LOCALONLY=1 (missing underscore) is silently ignored by extra="ignore", so
    # local_only stays False and the run proceeds to a public endpoint with NO output at all --
    # identical to a correctly-configured run that also printed nothing. A passing, ENFORCED
    # check has to leave different evidence than a silently-inert one.
    monkeypatch.delenv("MNEMIQ_LOCAL_ONLY", raising=False)
    _s(llm_base_url="http://127.0.0.1:8000/v1", llm_api_key="k").assert_local_only()
    err = capsys.readouterr().err
    assert "llm_base_url" in err


def test_a_disabled_check_prints_nothing(monkeypatch, capsys):
    monkeypatch.delenv("MNEMIQ_LOCAL_ONLY", raising=False)
    Settings(llm_base_url="https://api.openai.com/v1", llm_api_key="k").assert_local_only()
    assert capsys.readouterr().err == ""


# --- fix round 1, item 3: the verity_* endpoints are ordinary HTTP URLs ---------------------


def test_a_public_verity_endpoint_is_refused_and_named():
    # trace_sink POSTs to verity_traces_url on every ask/serve; even with the text/rows opt-ins
    # off, that body carries question hashes, timings, policy hashes and identity. Telemetry
    # leaving the data centre under a flag named LOCAL_ONLY is the false confidence this control
    # exists to remove.
    s = _s(
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_api_key="k",
        verity_traces_url="https://telemetry.example.com/api/traces/batch",
    )
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    assert "verity_traces_url" in str(exc.value)


def test_local_verity_endpoints_are_allowed():
    _s(
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_api_key="k",
        verity_records_url="http://10.0.0.5/api/semantic/records/open",
        verity_token_url="http://10.0.0.5/realms/x/protocol/openid-connect/token",
        verity_traces_url="http://10.0.0.5/api/traces/batch",
    ).assert_local_only()


# --- fix round 1, item 4: pg_dsn/control_dsn must be named as NOT checked -------------------


def test_the_refusal_names_the_dsns_it_does_not_check():
    # libpq accepts keyword form (`host=... port=...`), multi-host URIs, unix sockets and
    # `service=` files; `urlparse` returns hostname=None for keyword form, so a fail-closed
    # check would refuse every legitimate keyword DSN and silently miss the second host of a
    # multi-host URI. Rather than pretend to check these, the message has to say it does not.
    s = _s(llm_base_url="https://api.openai.com/v1", llm_api_key="k")
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    msg = str(exc.value)
    assert "pg_dsn" in msg and "control_dsn" in msg
    assert "not checked" in msg.lower()


# --- fix round 1, item 5: say the remedy, and stop reading as an accusation -----------------


def test_the_refusal_names_the_remedy():
    s = _s(llm_base_url="https://api.openai.com/v1", llm_api_key="k")
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    msg = str(exc.value)
    assert "RFC1918" in msg or "loopback" in msg
    assert "MNEMIQ_LOCAL_ONLY" in msg  # naming the escape hatch, not just the failure


def test_an_internal_dns_name_explains_rather_than_accuses():
    # `vllm.corp.internal` is a perfectly reasonable name for a DC's own LLM host. The old
    # message -- "is a name, not a private address" -- reads as a rebuke of that choice. The
    # actual reason it is refused is that this check does not resolve names at all, which is
    # true of ANY name, internal or public, and the message should say that instead.
    s = _s(llm_base_url="http://vllm.corp.internal:8000/v1", llm_api_key="k")
    with pytest.raises(RuntimeError) as exc:
        s.assert_local_only()
    msg = str(exc.value)
    assert "does not resolve" in msg
    assert "not a private address" not in msg


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


# --- fix round 2, item 6: nothing keeps the checked-field tuple complete --------------------


def test_every_url_field_is_checked_or_explicitly_exempted():
    # Nothing in Settings' own definition forces a new `..._url` field to be added to
    # `_LOCAL_ONLY_CHECKED_URL_FIELDS` -- a future `verity_something_url` would otherwise be
    # silently exempt from MNEMIQ_LOCAL_ONLY while the run still prints "verified" for
    # everything it DID check, which looks identical to a genuinely clean run. This test is the
    # forcing function: adding a `_url` field without updating one of the two sets below fails it.
    from mnemiq.config import Settings, _LOCAL_ONLY_CHECKED_URL_FIELDS

    # Fields deliberately NOT checked, with why -- there are none today. A field would go here,
    # commented, only if checking it were actively wrong (the way pg_dsn/control_dsn are wrong to
    # check as URLs at all, were either of them ever renamed to end in `_url`).
    EXEMPTED: set[str] = set()

    all_url_fields = {name for name in Settings.model_fields if name.endswith("_url")}
    accounted_for = set(_LOCAL_ONLY_CHECKED_URL_FIELDS) | EXEMPTED
    unaccounted = all_url_fields - accounted_for
    assert not unaccounted, (
        f"new _url field(s) {unaccounted} are neither checked by assert_local_only "
        "(_LOCAL_ONLY_CHECKED_URL_FIELDS) nor in this test's own EXEMPTED set"
    )
