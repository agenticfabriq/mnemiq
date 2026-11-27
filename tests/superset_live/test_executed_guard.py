"""Live: mnemiq's check of what Superset would run (`executed.check_executed`), held against what a real Superset
7.0.0rc2 answers: its own dry run (`execute_sql` with `dry_run`) and the extension's own `table_rules`, with a row rule
in place.

The adapter makes this check on Oracle only, and Superset mode on Oracle is checked live against an Oracle database in
its own task. Here the database behind Superset is the live suite's PostgreSQL one, so the check reads what Superset
answers in PostgreSQL's dialect: that is the mechanism, proved against the real Superset rather than against fixtures.
`policy` carries west's Regular row rule (`setup_superset.py`), which Superset applies to west and not to analyst.
"""

from __future__ import annotations

import json

import pytest

from mnemiq.superset.client import DelegatedToken, SupersetMcp
from mnemiq.superset.executed import check_executed, executed_sql, tables_read
from mnemiq.superset.extension import table_rules_apply
from mnemiq.superset.names import RefusedStatement
from mnemiq.superset.payload import check_execute_response

pytestmark = pytest.mark.integration

SCHEMA = "public"
STATEMENT = "SELECT p.region, COUNT(*) AS n FROM policy p GROUP BY p.region"
# A table qualified by its name in another case than Superset's wrap aliases it by: PostgreSQL reads POLICY as policy,
# and Superset wraps the table under the name as written, quoted ("POLICY"), which PostgreSQL reads as another name.
STRANDED = "SELECT POLICY.region, COUNT(*) AS n FROM POLICY GROUP BY POLICY.region"


def observe(test: str, **facts) -> None:
    print(f"OBSERVED {test} {json.dumps(facts, default=str, sort_keys=True)}")


def _client(live, username: str) -> SupersetMcp:
    """An MCP client carrying an ask token for `username`, as a person's request through mnemiq does."""
    return SupersetMcp(live.mcp_url, DelegatedToken(live.token(username)), timeout_s=90)


def _request(live, sql: str, **more) -> dict:
    return {"database_id": live.database_id, "sql": sql, "schema": SCHEMA, "timeout": 30, **more}


def _executed(live, client: SupersetMcp, sql: str, **more) -> str:
    """The statement Superset's dry run says it would run for `sql`, asked as `client`'s person."""
    payload = client.call("execute_sql", _request(live, sql, dry_run=True, **more))
    check_execute_response(payload)
    return executed_sql(payload)


def _rules(live, client: SupersetMcp, sql: str) -> dict[str, bool | None]:
    """Whether a row rule applies to `client`'s person on each table `sql` reads, as the extension answers it."""
    read = tables_read(sql, dialect="postgres")
    answers = table_rules_apply(client, database_id=live.database_id, schema=SCHEMA,
                                tables=[name for _, _, name in read.values()])
    return {key: answers.get(name) for key, (_, _, name) in read.items()}


def test_supersets_own_wrap_passes_and_a_ruled_table_left_bare_is_refused(live):
    """west's dry run wraps `policy` in a subquery carrying west's rule, and `table_rules` says a rule applies to west
    on it: the check passes. analyst's dry run leaves `policy` bare, and `table_rules` says no rule applies to
    analyst: the check passes too. analyst's bare statement held to west's rules is refused, naming the table."""
    west, analyst = _client(live, "west"), _client(live, "analyst")
    ruled, bare = _executed(live, west, STATEMENT), _executed(live, analyst, STATEMENT)
    west_rules, analyst_rules = _rules(live, west, STATEMENT), _rules(live, analyst, STATEMENT)
    assert west_rules == {"policy": True} and analyst_rules == {"policy": False}, (west_rules, analyst_rules)
    assert ruled != bare and "FROM (" in " ".join(ruled.split()), ruled
    check_executed(STATEMENT, ruled, rules=west_rules, dialect="postgres")
    check_executed(STATEMENT, bare, rules=analyst_rules, dialect="postgres")
    with pytest.raises(RefusedStatement, match="could not confirm for policy"):
        check_executed(STATEMENT, bare, rules=west_rules, dialect="postgres")
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        check_executed(STATEMENT.replace("COUNT(*)", "COUNT(p.id)"), ruled, rules=west_rules, dialect="postgres")
    observe("wrap", west=" ".join(ruled.split()), analyst=" ".join(bare.split()), west_rules=west_rules,
            analyst_rules=analyst_rules)


def test_supersets_limit_is_excepted_only_where_one_is_asked(live):
    """A dry run asked with `limit` puts Superset's limit in place of the statement's: the check reads the two alike
    when the call asks that limit, and refuses them when it asks none or fewer rows."""
    west = _client(live, "west")
    sql = f"{STATEMENT} ORDER BY p.region LIMIT 100"
    limited = _executed(live, west, sql, limit=5)
    assert "LIMIT 5" in limited, limited
    rules = _rules(live, west, sql)
    check_executed(sql, limited, rules=rules, dialect="postgres", limit=5)
    with pytest.raises(RefusedStatement, match="Superset would run a different statement"):
        check_executed(sql, limited, rules=rules, dialect="postgres")
    with pytest.raises(RefusedStatement, match="row limit"):
        check_executed(sql, limited, rules=rules, dialect="postgres", limit=4)
    observe("limit", limited=" ".join(limited.split()))


def test_a_qualifier_supersets_wrap_strands_is_refused(live):
    """For west, Superset wraps `POLICY` under `"POLICY"`, so the qualifier POLICY, which PostgreSQL reads as policy,
    names no source of the query: the check refuses the statement. Run as Superset would run it, PostgreSQL itself
    rejects it, so the refusal is no false alarm; for analyst, whom no rule narrows, the statement passes and runs."""
    west, analyst = _client(live, "west"), _client(live, "analyst")
    ruled = _executed(live, west, STRANDED)
    assert '"POLICY"' in ruled, ruled
    with pytest.raises(RefusedStatement, match=r"qualified by a name that is no one table or subquery"):
        check_executed(STRANDED, ruled, rules=_rules(live, west, STRANDED), dialect="postgres")
    ran = west.call("execute_sql", _request(live, STRANDED))
    assert ran.get("success") is False, ran
    bare = _executed(live, analyst, STRANDED)
    check_executed(STRANDED, bare, rules=_rules(live, analyst, STRANDED), dialect="postgres")
    assert analyst.call("execute_sql", _request(live, STRANDED)).get("success") is True
    observe("stranded", west=" ".join(ruled.split()), west_ran=ran.get("error_type"))
