"""**M106.** Every Verity call refuses a redirect instead of following it.

urllib follows a 3xx by rebuilding the request with every header but the content ones
(`HTTPRedirectHandler.redirect_request`), so a records pull or a trace post answered with a
redirect carried its `Authorization: Bearer` token to whatever host the redirect named, and a token
request answered with one took its token from that host instead of the configured issuer. Real
sockets here -- a fake Verity that answers 302 and an elsewhere host that records what reaches
it -- because a stubbed `urlopen` never follows a redirect, so it could not show this.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


class _Host:
    """A local HTTP host. With `redirect_to` it answers every request 302 to that base's
    `/redirected` -- a fixed path, never the request's own, so no header is built from what a
    client sent; without, 200 and `body` as JSON. `seen` records what reached it."""

    def __init__(self, body=None, redirect_to=None):
        self.seen: list[dict] = []
        self.body = body if body is not None else {}
        self.redirect_to = redirect_to
        host = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("content-length") or 0)
                sent = self.rfile.read(length) if length else b""
                host.seen.append({"method": self.command, "path": self.path,
                                  "authorization": self.headers.get("authorization"),
                                  "body": sent.decode(errors="replace")})
                if host.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", host.redirect_to + "/redirected")
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                payload = json.dumps(host.body).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _answer
            do_POST = _answer

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def hosts(monkeypatch):
    # A proxy from the environment would carry these requests somewhere else entirely.
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    made: list[_Host] = []

    def make(**kw):
        made.append(_Host(**kw))
        return made[-1]

    yield make
    for host in made:
        host.close()


@pytest.fixture(autouse=True)
def _fresh_tokens():
    from mnemiq.enrichment import verity_auth

    verity_auth.reset_token_cache()
    yield
    verity_auth.reset_token_cache()


def _credentials(issuer):
    return {"verity_token_url": f"{issuer.url}/token", "verity_client_id": "mnemiq",
            "verity_client_secret": "s3cret"}


def test_a_records_pull_redirected_elsewhere_carries_the_token_nowhere(hosts, tmp_path):
    from mnemiq.config import Settings
    from mnemiq.enrichment import certified

    issuer = hosts(body={"access_token": "verity-token", "expires_in": 300})
    elsewhere = hosts(body={"records": [], "watermark": "t1", "next_cursor": None})
    verity = hosts(redirect_to=elsewhere.url)
    settings = Settings(verity_records_url=f"{verity.url}/api/semantic/records/open",
                        verity_watermark_path=str(tmp_path / "wm.json"), **_credentials(issuer))

    got = certified.fetch_certified_records(settings)

    assert verity.seen and verity.seen[0]["authorization"] == "Bearer verity-token", (
        "setup: the pull reached Verity with its token")
    assert elsewhere.seen == [], f"the redirect was followed: {elsewhere.seen}"
    assert got.available is False, "a refused redirect is a pull that did not complete"


def test_a_trace_post_redirected_elsewhere_carries_the_token_nowhere(hosts):
    from mnemiq.observability.trace_sink import VerityTraceSink
    from tests.test_verity_trace_sink import _event, _Settings

    issuer = hosts(body={"access_token": "verity-token", "expires_in": 300})
    elsewhere = hosts(body={})
    verity = hosts(redirect_to=elsewhere.url)
    sink = VerityTraceSink(_Settings(verity_traces_url=f"{verity.url}/api/traces/batch",
                                     **_credentials(issuer)))

    sink.record_answer(_event())

    assert verity.seen and verity.seen[0]["authorization"] == "Bearer verity-token", (
        "setup: the trace reached Verity with its token")
    assert elsewhere.seen == [], f"the redirect was followed: {elsewhere.seen}"
    # A redirect is the receiver telling us our URL is wrong: permanent until someone fixes the
    # configuration, so it counts with the rejections, not with an outage that heals itself.
    assert (sink.dropped, sink.dropped_rejected) == (1, 1)


def test_a_token_request_redirected_elsewhere_takes_no_token_from_there(hosts):
    from mnemiq.enrichment import verity_auth

    elsewhere = hosts(body={"access_token": "elsewhere-token", "expires_in": 300})
    issuer = hosts(redirect_to=elsewhere.url)

    token = verity_auth.access_token(SimpleNamespace(**_credentials(issuer)))

    assert issuer.seen, "setup: the token request reached the configured issuer"
    assert elsewhere.seen == [], f"the redirect was followed: {elsewhere.seen}"
    assert token is None, "a token from a host the issuer redirected to is not the issuer's"


def test_nothing_in_mnemiq_calls_urlopen_except_through_the_redirect_refusing_seam():
    """The ratchet: a new Verity call written with the stdlib's `urlopen` follows redirects again."""
    from pathlib import Path

    import mnemiq

    root = Path(mnemiq.__file__).parent
    callers = sorted(str(path.relative_to(root)) for path in root.rglob("*.py")
                     if "urllib.request.urlopen(" in path.read_text())
    assert callers == [], f"call verity_http.urlopen, which refuses redirects: {callers}"
