"""A tenant that certified nothing and a Verity that refused us are not the same state.

`fetch_certified_records` returned a bare list, so both were `[]` and no caller could tell them
apart. That is M2's defect a second time: `GrantSet.available` exists because "the policy could not
be READ" and "the policy grants nothing" both denied everything and were the same empty set, and
its comment says an outage "is an outage an operator must be told about".

Certified records are worse, because the failure is silent in the direction that looks healthy. A
deployment configured to answer from a certified corpus, whose pull 401s, answers UNGROUNDED and
looks exactly like one that is working. Beacon guards its own benchmark against this with an
`expect_records` gate for exactly this reason; nothing guarded the engine.
"""

import io
import json as _json
import urllib.error


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _record(object_id):
    return {
        "envelope": {"object_type": "definition", "object_id": object_id, "version": "v1",
                     "source_system": "pg"},
        "payload": {"id": object_id, "term": object_id, "domain": "ops", "definition": "d"},
    }


def _settings(tmp_path, **extra):
    from mnemiq.config import Settings
    return Settings(verity_records_url="https://v/api/semantic/records/open",
                    verity_watermark_path=str(tmp_path / "wm.json"), **extra)


def test_a_tenant_with_nothing_certified_is_available(tmp_path, monkeypatch):
    """The legitimate empty state. Certifying nothing yet is a real deployment, and erroring on it
    would block one -- the error is on FAILURE, never on emptiness."""
    from mnemiq.enrichment import certified as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(_json.dumps(
                            {"records": [], "watermark": "t1", "next_cursor": None}).encode()))
    got = mod.fetch_certified_records(_settings(tmp_path))
    assert got.records == []
    assert got.available is True, "we asked and were answered; there is simply nothing certified"


def test_a_refused_pull_with_a_cold_cache_is_not_available(tmp_path, monkeypatch):
    """The failure state, and the one that looks identical from the outside."""
    from mnemiq.enrichment import certified as mod

    def refuse(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", refuse)
    got = mod.fetch_certified_records(_settings(tmp_path))
    assert got.records == []
    assert got.available is False, "we could not read the corpus; that is an outage, not an answer"


def test_the_two_states_are_distinguishable(tmp_path, monkeypatch):
    """The whole point, stated as the comparison that used to be impossible."""
    from mnemiq.enrichment import certified as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(_json.dumps(
                            {"records": [], "watermark": "t1", "next_cursor": None}).encode()))
    empty = mod.fetch_certified_records(_settings(tmp_path))

    def refuse(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", refuse)
    refused = mod.fetch_certified_records(_settings(tmp_path / "cold"))

    assert empty.records == refused.records == []
    assert empty.available != refused.available, (
        "identical record lists; only the flag separates a legitimate empty corpus from an outage"
    )


def test_a_served_cache_is_available(tmp_path, monkeypatch):
    """Last-known-good is an ANSWER, not an outage. A failed page that falls back to a populated
    cache still grounds the engine, so it must not trip the refusal -- M18's degrade-to-cached is
    the behaviour being preserved, not overridden."""
    from mnemiq.enrichment import certified as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(_json.dumps(
                            {"records": [_record("a")], "watermark": "t1",
                             "next_cursor": None}).encode()))
    settings = _settings(tmp_path)
    assert mod.fetch_certified_records(settings).records, "seed the cache"

    def refuse(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", refuse)
    got = mod.fetch_certified_records(settings)
    assert [r.envelope.object_id for r in got.records] == ["a"]
    assert got.available is True, "we are serving a real corpus; degraded freshness is not absence"


def test_no_url_configured_is_available(tmp_path):
    """Unconfigured is not a failure. A local run with no Verity is a supported deployment and must
    not be refused -- the refusal is for a deployment that ASKED for a corpus and did not get
    one."""
    from mnemiq.config import Settings
    from mnemiq.enrichment import certified as mod

    got = mod.fetch_certified_records(Settings())
    assert got.records == []
    assert got.available is True


def test_a_configured_deployment_refuses_rather_than_grounding_on_nothing(tmp_path):
    """The decision this flag exists to enable, stated as the refusal.

    Fail-soft is right for freshness and wrong for grounding. A deployment that asked for a
    certified corpus and cannot read it composes every answer without the meanings a human
    certified -- and looks exactly like a healthy ungrounded engine, which is why nothing caught
    it. The eval path is the sharpest case: its own comment said "fetch is fail-soft: no
    verity_records_url -> [] -> no-op", true of the UNCONFIGURED case it was written about and
    silently also covering the 401, so a benchmark run could measure an ungrounded engine and
    report it as the grounded arm.
    """
    import pytest

    from mnemiq.config import Settings
    from mnemiq.enrichment.certified import CertifiedSet, require_certified

    configured = Settings(verity_records_url="https://v/api/semantic/records/open")
    with pytest.raises(RuntimeError, match="could not be read"):
        require_certified(CertifiedSet([], available=False), configured)


def test_the_refusal_fires_only_on_failure_never_on_emptiness(tmp_path):
    """Three states that must NOT refuse, because each is a real deployment: a tenant that has
    certified nothing, a local run that never configured Verity, and a failed pull that degraded to
    a populated cache. Erroring on any of them would trade one silent wrong answer for a loud
    wrong refusal."""
    from mnemiq.config import Settings
    from mnemiq.enrichment.certified import CertifiedSet, require_certified

    configured = Settings(verity_records_url="https://v/api/semantic/records/open")
    require_certified(CertifiedSet([], available=True), configured)          # nothing certified yet
    require_certified(CertifiedSet(["r"], available=True), configured)       # cache served
    require_certified(CertifiedSet([], available=False), Settings())         # never configured


def test_every_fetch_site_also_requires():
    """A fetch without the guard is the bug this change fixes, reintroduced by a fifth call site.

    Presence of the name is not the check -- `require_certified` appears in each of these files
    twice, once as an import, and a file that imports it and never calls it would satisfy a grep.
    So this asserts CALLS, matched by AST, and pins the exact set: a new fetch site fails here and
    has to decide what an unreadable corpus means for it, which is the decision the flag exists to
    force.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "mnemiq"
    assert root.is_dir(), "the scan must have something to walk"

    fetches, guards = set(), set()
    for path in root.rglob("*.py"):
        rel = str(path.relative_to(root))
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "fetch_certified_records":
                fetches.add(rel)
            elif name == "require_certified":
                guards.add(rel)

    assert fetches == {"cli.py", "eval/run.py", "eval/bird_runner.py", "eval/engine.py"}, (
        f"the set of fetch sites changed: {sorted(fetches)}. Add it here AND guard it"
    )
    assert fetches <= guards, (
        f"these fetch a certified corpus and never check it is readable: {sorted(fetches - guards)}"
    )


def test_the_set_has_no_truth_value():
    """`if certified:` used to mean "non-empty". On an object it is silently True even for a corpus
    we could not read -- the same conflation this type exists to remove, wearing a different hat.
    So it raises rather than answering, which is the non-iterable decision applied to the other
    implicit conversion."""
    import pytest

    from mnemiq.enrichment.certified import CertifiedSet

    with pytest.raises(TypeError, match="no truth value"):
        bool(CertifiedSet([], available=False))
    with pytest.raises(TypeError):
        if CertifiedSet(["r"], available=True):  # the point is that this raises
            pass
