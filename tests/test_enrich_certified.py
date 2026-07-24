from mnemiq.contract import (
    CertifiedRecord, CodedValue, CodeScheme, Column, Definition, RecordEnvelope, Snapshot,
)


def _env(object_type, object_id):
    return RecordEnvelope(object_type=object_type, object_id=object_id,
                          version="v1", source_system="pg")


def _local_snapshot():
    # a locally-profiled column: has structure (data_type/stats), no meaning yet
    return Snapshot(version="v", source_id="s", created_at="t", columns=[
        Column(id="orders.status", object_id="orders", name="status", data_type="text",
               distinct_count=3, row_count=100,
               coded_values=[CodedValue(code="N"), CodedValue(code="S")])])


def test_column_merge_takes_meaning_from_cert_and_structure_from_local():
    from mnemiq.enrichment.certified import apply_certified

    cert_col = Column(id="orders.status", object_id="orders", name="status",
                      description="Order status.", semantic_type="category",
                      coded_values=[CodedValue(code="N", meaning="New"),
                                    CodedValue(code="S", meaning="Shipped")],
                      code_scheme=CodeScheme(id="urn:status", label="Order Status"))
    out = apply_certified(_local_snapshot(),
                          [CertifiedRecord(envelope=_env("column", "orders.status"), payload=cert_col)])
    col = next(c for c in out.columns if c.id == "orders.status")

    # meaning from cert
    assert col.description == "Order status."
    assert col.code_scheme.label == "Order Status"
    assert {cv.code: cv.meaning for cv in col.coded_values} == {"N": "New", "S": "Shipped"}
    assert all(cv.source == "certified" for cv in col.coded_values)
    # structure stays local
    assert col.data_type == "text"
    assert col.distinct_count == 3 and col.row_count == 100


def test_standalone_definition_record_is_added():
    from mnemiq.enrichment.certified import apply_certified

    d = Definition(id="def-status", term="status", domain="ops", definition="fulfilment state")
    out = apply_certified(_local_snapshot(),
                          [CertifiedRecord(envelope=_env("definition", "def-status"), payload=d)])
    assert [x.term for x in out.definitions] == ["status"]


def test_standalone_dimension_record_is_routed_into_snapshot():
    from mnemiq.enrichment.certified import apply_certified
    from mnemiq.contract import Dimension

    dim = Dimension(id="dim-region", label="Region", source="orders.region")
    out = apply_certified(
        _local_snapshot(),
        [CertifiedRecord(envelope=_env("dimension", "dim-region"), payload=dim)],
    )
    assert any(d.id == "dim-region" for d in out.dimensions)


def test_cert_for_a_column_not_in_the_db_is_skipped():
    from mnemiq.enrichment.certified import apply_certified

    ghost = Column(id="orders.ghost", object_id="orders", name="ghost", description="nope")
    out = apply_certified(_local_snapshot(),
                          [CertifiedRecord(envelope=_env("column", "orders.ghost"), payload=ghost)])
    assert not any(c.id == "orders.ghost" for c in out.columns)  # stale cert, not grounded
    assert next(c for c in out.columns if c.id == "orders.status").description is None


def _record_json():
    return {
        "envelope": {"object_type": "column", "object_id": "orders.status",
                     "version": "v1", "source_system": "pg"},
        "payload": {"id": "orders.status", "object_id": "orders", "name": "status",
                    "description": "Order status."},
    }


def test_fetch_parses_records_and_presents_the_bearer_token(monkeypatch):
    import io
    import json as _json

    from mnemiq.config import Settings
    from mnemiq.enrichment import certified as mod
    from mnemiq.enrichment import verity_auth

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    verity_auth.reset_token_cache()
    seen: list = []

    def fake_urlopen(req, timeout=0):
        seen.append(req)
        if req.full_url.endswith("/api/auth/token"):
            return _Resp(_json.dumps({"access_token": "tok-1", "expires_in": 600}).encode())
        return _Resp(_json.dumps({"records": [_record_json(),
                                              {"envelope": {}, "payload": {}}]}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    records = mod.fetch_certified_records(Settings(
        verity_records_url="https://v/api/semantic/records",
        verity_token_url="https://v/api/auth/token",
        verity_client_id="cid_abc123",
        verity_client_secret="s3cret"))

    assert len(records) == 1  # the malformed second item is skipped, not fatal
    assert records[0].payload.description == "Order status."
    records_request = seen[-1]
    assert records_request.get_header("Authorization") == "Bearer tok-1"


def test_fetch_sends_no_authorization_header_when_unconfigured(monkeypatch):
    import io
    import json as _json

    from mnemiq.config import Settings
    from mnemiq.enrichment import certified as mod
    from mnemiq.enrichment import verity_auth

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    verity_auth.reset_token_cache()
    seen: list = []

    def fake_urlopen(req, timeout=0):
        seen.append(req)
        return _Resp(_json.dumps({"records": [_record_json()]}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    records = mod.fetch_certified_records(
        Settings(verity_records_url="https://v/api/semantic/records"))

    assert len(records) == 1
    assert len(seen) == 1, "no token exchange without client credentials"
    assert seen[0].get_header("Authorization") is None


def test_fetch_refreshes_the_token_once_on_401(monkeypatch):
    import io
    import json as _json
    import urllib.error

    from mnemiq.config import Settings
    from mnemiq.enrichment import certified as mod
    from mnemiq.enrichment import verity_auth

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    verity_auth.reset_token_cache()
    minted: list = []
    record_attempts: list = []

    def fake_urlopen(req, timeout=0):
        if req.full_url.endswith("/api/auth/token"):
            minted.append(req)
            return _Resp(_json.dumps(
                {"access_token": f"tok-{len(minted)}", "expires_in": 600}).encode())
        record_attempts.append(req.get_header("Authorization"))
        if len(record_attempts) == 1:  # the cached token was revoked/expired server-side
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
        return _Resp(_json.dumps({"records": [_record_json()]}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    records = mod.fetch_certified_records(Settings(
        verity_records_url="https://v/api/semantic/records",
        verity_token_url="https://v/api/auth/token",
        verity_client_id="cid_abc123",
        verity_client_secret="s3cret"))

    assert len(records) == 1, "the retry after refresh must succeed"
    assert record_attempts == ["Bearer tok-1", "Bearer tok-2"]
    assert len(minted) == 2, "exactly one refresh"


def test_fetch_gives_up_after_one_refresh_when_401_persists(monkeypatch):
    import io
    import json as _json
    import urllib.error

    from mnemiq.config import Settings
    from mnemiq.enrichment import certified as mod
    from mnemiq.enrichment import verity_auth

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    verity_auth.reset_token_cache()
    record_attempts: list = []

    def fake_urlopen(req, timeout=0):
        if req.full_url.endswith("/api/auth/token"):
            return _Resp(_json.dumps({"access_token": "tok", "expires_in": 600}).encode())
        record_attempts.append(req)
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    # A revoked credential 401s forever: degrade to local-only rather than loop.
    assert mod.fetch_certified_records(Settings(
        verity_records_url="https://v/api/semantic/records",
        verity_token_url="https://v/api/auth/token",
        verity_client_id="cid_abc123",
        verity_client_secret="s3cret")) == []
    assert len(record_attempts) == 2


def test_fetch_is_fail_soft_on_network_error(monkeypatch):
    import urllib.error

    from mnemiq.config import Settings
    from mnemiq.enrichment import certified as mod

    def boom(req, timeout=0):
        raise urllib.error.URLError("verity down")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    assert mod.fetch_certified_records(
        Settings(verity_records_url="https://v/api/semantic/records")) == []


def test_fetch_returns_empty_when_unconfigured():
    from mnemiq.config import Settings
    from mnemiq.enrichment.certified import fetch_certified_records

    assert fetch_certified_records(Settings()) == []


def test_flow_certified_over_local_but_dictionary_over_certified():
    """The precedence blend end to end: local grounding fills a code, certified overrides it,
    the operator dictionary overrides certified."""
    from mnemiq.contract import (
        CertifiedRecord, CodedValue, Column, RecordEnvelope, Snapshot,
    )
    from mnemiq.enrichment.certified import apply_certified
    from mnemiq.enrichment.dictionary import ColumnEntry, DataDictionary
    from mnemiq.enrichment.grounding import apply_dictionary

    snap = Snapshot(version="v", source_id="s", created_at="t", columns=[
        Column(id="orders.status", object_id="orders", name="status", data_type="text",
               coded_values=[CodedValue(code="N", meaning="local-new", source="lookup")])])

    cert = Column(id="orders.status", object_id="orders", name="status",
                  coded_values=[CodedValue(code="N", meaning="cert-new")])
    snap = apply_certified(
        snap, [CertifiedRecord(envelope=RecordEnvelope(
            object_type="column", object_id="orders.status", version="v", source_system="pg"),
            payload=cert)])
    assert snap.columns[0].coded_values[0].meaning == "cert-new"      # certified beat lookup
    assert snap.columns[0].coded_values[0].source == "certified"

    snap = apply_dictionary(snap, DataDictionary(columns={
        "orders.status": ColumnEntry(codes={"N": "dict-new"})}))
    assert snap.columns[0].coded_values[0].meaning == "dict-new"      # dictionary beat certified
    assert snap.columns[0].coded_values[0].source == "dictionary"
