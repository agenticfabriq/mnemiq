"""What several Superset-mode test modules share: a deployment in Superset mode over a store holding one snapshot,
Superset's own words and mnemiq's sentences the tests read answers by, the texts rc2's MCP saves as other text, and the
one-chart build plan.

Defined once here and imported by each module that uses them (`test_superset_runtime.py`, `test_superset_http.py`,
`test_superset_builder.py`, `test_superset_build_http.py`, `test_superset_planner.py`, `test_superset_measures.py`),
so no test module imports another.
"""

from __future__ import annotations

import json

from mnemiq.config import Settings
from mnemiq.contract import Column, Snapshot
from mnemiq.contract.semantic import Job, SourceBinding
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import save_snapshot
from mnemiq.superset.charts import ChartSpec, Family, WrittenRef
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.planner import BuildPlan
from mnemiq.superset.settings import SUPERSET_SOURCE_JOB
from tests.superset_build_world import POLICY_DATASET

# --- a Superset-mode deployment

DATABASE_ID = 7
SCHEMA = "public"
SOURCE_ID = "only"


def _source_job(*, status: str = "done", detail: str | None = None, **fields) -> Job:
    """The job a snapshot enriched through Superset carries, naming the database and schema it was read from."""
    values = {"database_id": DATABASE_ID, "schema": SCHEMA, "backend": "postgresql", "values": "not harvested"}
    values.update(fields)
    return Job(id=SUPERSET_SOURCE_JOB, source_id=SOURCE_ID, kind="superset", status=status,
               detail=json.dumps(values) if detail is None else detail)


def _snapshot(jobs: list[Job], version: str = "v1") -> Snapshot:
    return Snapshot(version=version, source_id=SOURCE_ID, created_at="t",
                    columns=[Column(id="wide.col_0", object_id="wide", name="col_0", data_type="VARCHAR")],
                    source_bindings=[SourceBinding(id="sb:wide", source_id=SOURCE_ID, object_id="wide",
                                                   source_object="wide", binding_type="table")],
                    jobs=jobs)


def superset_deployment(tmp_path, model_url: str, superset_url: str, *, jobs: list[Job] | None = None,
                        snapshot: bool = True, profiles: str | None = None, **settings) -> Settings:
    """Settings for a Superset-mode deployment over a store whose snapshot carries `jobs` (by default the Superset
    source job, as enrichment through Superset writes it), with `profiles` as its profiles file when given."""
    store = tmp_path / "store.duckdb"
    con = init_store(str(store))
    if snapshot:
        save_snapshot(con, _snapshot([_source_job()] if jobs is None else jobs))
    con.close()
    if profiles is not None:
        (tmp_path / "profiles.toml").write_text(profiles)
        settings["profiles_path"] = str(tmp_path / "profiles.toml")
    return Settings(acme_data_dir=None, llm_base_url=model_url, llm_api_key="stub-key", llm_model="stub-model",
                    store_path=str(store), source_id=SOURCE_ID, answer_log_path="off",
                    llm_window_check=settings.pop("llm_window_check", False), superset_mcp_url=superset_url,
                    superset_database_id=DATABASE_ID, superset_schema=SCHEMA, **settings)


# --- the words answers are read by

SIGN_IN = "Superset did not accept this sign-in; sign in to Superset again."
SNAPSHOT_REFUSED = ("This deployment's knowledge is not the one enriched for the Superset database it answers from, so "
                    "nothing is answered until an operator enriches it through Superset; mnemiq's log records why.")
# Superset's own text for a role that does not permit SQL, as its MCP service renders it (`superset/mcp_service/auth.py`
# and `superset/mcp_service/middleware.py`, `GlobalErrorHandlerMiddleware`).
ROLE_DENIAL = "Permission denied: can_execute_sql_query on SQLLab for user west (tool: execute_sql)"


# --- what rc2's MCP does to a chart's name and to a SQL metric's label

# rc2's MCP runs every chart name (`GenerateChartRequest.sanitize_chart_name`) and every SQL metric's label
# (`ColumnRef.sanitize_label`, `superset/mcp_service/chart/schemas.py`) through `sanitize_user_input`
# (`superset/mcp_service/utils/sanitization.py`): it strips the ends, removes zero-width, control and line-separator
# characters, unescapes entities and cleans out HTML with nh3 (keeping only `&amp;` as `&`), and refuses a
# `javascript:`, `vbscript:` or `data:` scheme and an `on...=` handler. A text it changes is stored as some other text,
# which the build then cannot find again by its name, nor its value's format by its label; one it refuses makes no
# chart.
CHANGED_OR_REFUSED_BY_RC2 = {
    "a tag": "Premium <b>by</b> region", "an angle bracket": "Premium > 100", "a less-than sign": "Premium < 1y",
    "an ampersand": "Premium & claims", "an entity": "Premium &amp; claims",
    "a zero-width space": "Premium\u200bby region", "a byte order mark": "Premium\ufeff",
    "a control character": "Premium\x07 by region", "a tab": "Premium\tby region",
    "a line separator": "Premium\u2028by region", "a no-break space": "Premium\u00a0by region",
    "a script scheme": "javascript:alert", "a data scheme": "Data: by region", "an event handler": "monthly = target",
    "an event handler, spaced": "Premium ONLOAD = 1", "a space at its end": "Premium by region ",
}
KEPT_BY_RC2 = ["Premium by region (USD)", "Claims — 2026", "Premium / policy", 'The "core" book', "50% share",
               "Naïve café totals", "On target", "Premium, on time"]


# --- a one-chart build plan, on the policy table's dataset of `tests/superset_build_world.py`

POLICIES = Measure(name="policies", label="Policies", expression="COUNT(policy_id)", kind=MeasureKind.COUNT,
                   dataset_id=POLICY_DATASET, description="Policies written", asked_by="")


def _chart(key: str, family: Family, title: str, dataset_id: int, measures: tuple, *, dimensions: tuple = (),
           time_column: str | None = None, time_grain: str | None = None) -> ChartSpec:
    return ChartSpec(key=key, family=family, title=title, dataset_id=dataset_id, measures=measures,
                     dimensions=dimensions, time_column=time_column, time_grain=time_grain, filters=(),
                     row_limit=None, source=None, target=None)


BY_REGION = _chart("policies_by_region", Family.BAR, "Policies by region", POLICY_DATASET, (WrittenRef(POLICIES),),
                   dimensions=("region",))
ONE_BAR = BuildPlan(charts=(BY_REGION,), measures=(POLICIES,))
