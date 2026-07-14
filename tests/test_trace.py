from mnemiq.agent.trace import build_trace
from mnemiq.contract import IdentityContext, Trace
from mnemiq.sql.verdict import Approved


def _approved() -> Approved:
    return Approved(
        plan_sql="SELECT count(*) AS n FROM claim LIMIT 1000",
        target_sql="SELECT count(*) AS n FROM claim LIMIT 1000",
        tables=["claim"],
        columns=["n"],
    )


def _identity() -> IdentityContext:
    return IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def test_the_trace_records_what_the_engine_actually_did():
    trace = build_trace(
        question="how many claims?",
        approved=_approved(),
        identity=_identity(),
        enrichment_version="v1",
        timing={"execute_ms": 12.0, "total_ms": 900.0},
        result_shape="scalar",
    )

    assert isinstance(trace, Trace)
    assert trace.question == "how many claims?"
    assert trace.plan_sql.startswith("SELECT")
    assert trace.target_sql.startswith("SELECT")
    assert trace.tables_used == ["claim"]
    assert trace.enrichment_version == "v1"
    assert trace.identity.principal_id == "u1"
    assert trace.timing["execute_ms"] == 12.0


def test_the_trace_round_trips_as_the_open_contract():
    trace = build_trace(
        question="q",
        approved=_approved(),
        identity=_identity(),
        enrichment_version="v1",
        timing={"total_ms": 1.0},
        result_shape="table",
    )
    assert Trace.model_validate_json(trace.model_dump_json()) == trace
