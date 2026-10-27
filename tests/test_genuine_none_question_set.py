"""M146's pre-registered measurement: 20 questions whose true answer is "none", and 20 controls.

The register named the missing evidence: no corpus in hand had a genuine "none". BIRD's five saved
SQLite runs hold no empty gold in 487 cases, and the two in its Postgres port presuppose a row. So
this set is built on the repo's own synthetic data -- the quickstart's demo database
(`scripts/seed_demo.py`), four invented tables -- and written, with each case's expected outcome,
before the rule it measures was run against it.

- **20 genuine "none" questions.** The SQL is the right SQL and the true answer is empty (or, for
  a total or a maximum, one NULL; one is a count, which was answered before). Asserted against the
  data, not assumed. The expected outcome is pre-registered per case and per engine, including the
  honest residual: a "none" filtered on a value the index does not cover stays declined, as does a
  text date range where the column holds text.
- **20 misspelled-value controls.** A wrong value -- a typo, a case slip, a missing accent, a date
  in the wrong format -- whose intended spelling has rows (asserted), so "none" would be a wrong
  answer. Every one must stay declined: before the run where the index covers the column (`check_
  values`), after it where nothing could check the value.

Two engines, both through the product's own adapter: the demo file as `mnemiq` attaches it (a
SQLite source read through DuckDB, its dates TEXT), and the same rows in a DuckDB file whose dates
are DATE. `instant` mode, so the deterministic rule is the whole decision; the model calls are
scripted to return each case's SQL. A count of 1 of 20 "none" questions answered is the result
before M146, measured on the parent commit with this same file.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.modes import MODES, build_agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.generate.generator import FakeGenerator
from mnemiq.runtime import _build_mode_verifiers
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.semantic.values import ValueIndex, build_value_index
from mnemiq.store.bootstrap import init_store
from scripts import seed_demo

ENGINES = ("demo", "typed")
A, D = "answered", "declined"
_J = ("FROM order_line ol JOIN customer c ON c.customer_id = ol.customer_id "
      "JOIN product p ON p.product_id = ol.product_id")


@dataclass(frozen=True)
class Case:
    id: str
    kind: str  # "none" | "control"
    question: str
    sql: str
    expect: dict  # engine -> A | D, pre-registered
    fix: dict | None = None  # a control's misspelling -> its intended spelling, which has rows


def _both(outcome: str) -> dict:
    return {"demo": outcome, "typed": outcome}


CASES = [
    # --- genuine "none": every filter value grounded, or numbers alone ---------------------------
    Case("N01", "none", "Which customers in Germany are in segment A?",
         "SELECT name FROM customer WHERE country = 'Germany' AND segment_cd = 'A'", _both(A)),
    Case("N02", "none", "Which French customers are in segment C?",
         "SELECT name FROM customer WHERE country = 'France' AND segment_cd = 'C'", _both(A)),
    Case("N03", "none", "List the order lines with a quantity over 100.",
         "SELECT order_id, quantity FROM order_line WHERE quantity > 100", _both(A)),
    Case("N04", "none", "Which order lines had a quantity of exactly 13?",
         "SELECT order_id, product_id FROM order_line WHERE quantity = 13", _both(A)),
    Case("N05", "none", "Which Dairy products did Netherlands customers order more than 20 of?",
         f"SELECT p.name, ol.quantity {_J} WHERE c.country = 'Netherlands' "
         "AND p.category = 'Dairy' AND ol.quantity > 20", _both(A)),
    Case("N06", "none", "Which Pantry products did German customers order?",
         f"SELECT DISTINCT p.name {_J} WHERE c.country = 'Germany' AND p.category = 'Pantry'",
         _both(A)),
    Case("N07", "none", "What is the total quantity of Pantry products ordered by French customers?",
         f"SELECT sum(ol.quantity) AS total {_J} WHERE c.country = 'France' "
         "AND p.category = 'Pantry'", _both(A)),
    Case("N08", "none", "What is the largest Pantry order quantity from segment C customers?",
         f"SELECT max(ol.quantity) AS largest {_J} WHERE c.segment_cd = 'C' "
         "AND p.category = 'Pantry'", _both(A)),
    Case("N09", "none", "Which United States customers are in segment B?",
         "SELECT name FROM customer WHERE country = 'United States' AND segment_cd = 'B'",
         _both(A)),
    Case("N10", "none", "Which customers in Germany or the United States are in segment B?",
         "SELECT name FROM customer WHERE country IN ('Germany', 'United States') "
         "AND segment_cd = 'B'", _both(A)),
    Case("N11", "none", "Which Dry goods order lines had a quantity below 2?",
         "SELECT ol.order_id FROM order_line ol JOIN product p ON p.product_id = ol.product_id "
         "WHERE p.category = 'Dry goods' AND ol.quantity < 2", _both(A)),
    # --- genuine "none" on dates: a bound on a DATE column; text where the column holds text -----
    Case("N12", "none", "Which orders were placed after 2024?",
         "SELECT DISTINCT order_id FROM order_line WHERE order_date > '2024-12-31'",
         {"demo": D, "typed": A}),
    Case("N13", "none", "Which customers signed up in 2023?",
         "SELECT name FROM customer WHERE signup_date BETWEEN '2023-01-01' AND '2023-12-31'",
         {"demo": D, "typed": A}),
    # On the demo's text dates the column is indexed, so a date no order has is refused before
    # the run as a value that does not appear: the residual of reading a date as a code.
    Case("N14", "none", "Which orders were placed on 3 August 2024?",
         "SELECT DISTINCT order_id FROM order_line WHERE order_date = '2024-08-03'",
         {"demo": D, "typed": A}),
    # --- genuine "none" through a CTE and subqueries: the resolver places the column -------------
    Case("N15", "none", "Which customers with more than two order lines are in Germany?",
         "WITH busy AS (SELECT customer_id FROM order_line GROUP BY customer_id "
         "HAVING count(*) > 2) SELECT c.name FROM customer c JOIN busy b "
         "ON b.customer_id = c.customer_id WHERE c.country = 'Germany'", _both(A)),
    Case("N16", "none", "Which German customers have an id above 7?",
         "WITH c AS (SELECT customer_id, country AS nation FROM customer) "
         "SELECT customer_id FROM c WHERE nation = 'Germany' AND customer_id > 7", _both(A)),
    Case("N17", "none", "Which products did segment C customers order more than 25 of?",
         "SELECT name FROM product WHERE product_id IN (SELECT product_id FROM order_line "
         "WHERE quantity > 25 AND customer_id IN (SELECT customer_id FROM customer "
         "WHERE segment_cd = 'C'))", _both(A)),
    # A count was answered before: a "none" asked as a count is a 0.
    Case("N18", "none", "How many customers in Germany are in segment B?",
         "SELECT count(*) AS n FROM customer WHERE country = 'Germany' AND segment_cd = 'B'",
         _both(A)),
    # --- genuine "none" on a value the index does not cover: the honest residual -----------------
    Case("N19", "none", "Which Dry goods did Harrow Provisions order?",
         f"SELECT p.name {_J} WHERE c.name = 'Harrow Provisions' AND p.category = 'Dry goods'",
         _both(D)),
    Case("N20", "none", "Which Enterprise-segment customers are in Germany?",
         "SELECT c.name FROM customer c JOIN segment_lookup s ON s.segment_cd = c.segment_cd "
         "WHERE s.segment_name = 'Enterprise' AND c.country = 'Germany'", _both(D)),
    # --- controls: a misspelled value whose intended spelling has rows ---------------------------
    Case("C01", "control", "Which orders did Aurora Bakehouse place?",
         "SELECT ol.order_id FROM order_line ol JOIN customer c ON c.customer_id = ol.customer_id "
         "WHERE c.name = 'Aurora Bakehose'", _both(D), {"Aurora Bakehose": "Aurora Bakehouse"}),
    Case("C02", "control", "Which orders did Northwind Grocers place?",
         "SELECT ol.order_id FROM order_line ol JOIN customer c ON c.customer_id = ol.customer_id "
         "WHERE c.name = 'Northwind Grocer'", _both(D), {"Northwind Grocer": "Northwind Grocers"}),
    Case("C03", "control", "Which orders included cultured butter?",
         "SELECT ol.order_id FROM order_line ol JOIN product p ON p.product_id = ol.product_id "
         "WHERE p.name = 'Cultured buttter'", _both(D), {"Cultured buttter": "Cultured butter"}),
    Case("C04", "control", "Which orders included crème fraîche?",
         "SELECT ol.order_id FROM order_line ol JOIN product p ON p.product_id = ol.product_id "
         "WHERE p.name = 'Creme fraiche'", _both(D), {"Creme fraiche": "Crème fraîche"}),
    Case("C05", "control", "Which customers are in the Mid-market segment?",
         "SELECT c.name FROM customer c JOIN segment_lookup s ON s.segment_cd = c.segment_cd "
         "WHERE s.segment_name = 'Midmarket'", _both(D), {"Midmarket": "Mid-market"}),
    Case("C06", "control", "Which customers are in the Small business segment?",
         "SELECT c.name FROM customer c JOIN segment_lookup s ON s.segment_cd = c.segment_cd "
         "WHERE s.segment_name = 'Small Business'", _both(D), {"Small Business": "Small business"}),
    Case("C07", "control", "Which customers have Bakehouse in their name?",
         "SELECT name FROM customer WHERE name LIKE '%Bakehose%'", _both(D),
         {"%Bakehose%": "%Bakehouse%"}),
    Case("C08", "control", "Which products start with Sourdough?",
         "SELECT name FROM product WHERE name LIKE 'Sour dough%'", _both(D),
         {"Sour dough%": "Sourdough%"}),
    # Indexed columns: refused before the run, naming the real values, never an empty answer.
    Case("C09", "control", "Which customers are in France?",
         "SELECT name FROM customer WHERE country = 'Frnace'", _both(D), {"Frnace": "France"}),
    Case("C10", "control", "Which customers are in Germany?",
         "SELECT name FROM customer WHERE country = 'germany'", _both(D), {"germany": "Germany"}),
    Case("C11", "control", "Which products are Dairy?",
         "SELECT name FROM product WHERE category = 'Dairy products'", _both(D),
         {"Dairy products": "Dairy"}),
    Case("C12", "control", "Which customers are in segment A?",
         "SELECT name FROM customer WHERE segment_cd = 'a'", _both(D), {"'a'": "'A'"}),
    # A date in the wrong format, against the date as text.
    Case("C13", "control", "Which orders were placed in September 2024?",
         "SELECT DISTINCT order_id FROM order_line "
         "WHERE CAST(order_date AS VARCHAR) LIKE '2024/09%'", _both(D), {"2024/09%": "2024-09%"}),
    Case("C14", "control", "Which orders were placed in September 2024?",
         "SELECT DISTINCT order_id FROM order_line "
         "WHERE substr(CAST(order_date AS VARCHAR), 1, 7) = '2024/09'", _both(D),
         {"2024/09": "2024-09"}),
    Case("C15", "control", "Which orders did Perch & Co or Vellum Supply place?",
         "SELECT ol.order_id FROM order_line ol JOIN customer c ON c.customer_id = ol.customer_id "
         "WHERE c.name IN ('Perch and Co', 'Vellum Suply')", _both(D),
         {"Perch and Co": "Perch & Co", "Vellum Suply": "Vellum Supply"}),
    Case("C16", "control", "Which orders did Cinder Lane Cafe place?",
         "SELECT ol.order_id FROM order_line ol JOIN customer c ON c.customer_id = ol.customer_id "
         "WHERE lower(c.name) = 'cinder lane café'", _both(D),
         {"cinder lane café": "cinder lane cafe"}),
    Case("C17", "control", "Which orders did Saltmarsh Trading place?",
         "WITH x AS (SELECT customer_id FROM customer WHERE name = 'Saltmarsh Tradng') "
         "SELECT order_id FROM order_line WHERE customer_id IN (SELECT customer_id FROM x)",
         _both(D), {"Saltmarsh Tradng": "Saltmarsh Trading"}),
    Case("C18", "control", "What total quantity did Meridian Foods order?",
         "SELECT sum(ol.quantity) AS total FROM order_line ol JOIN customer c "
         "ON c.customer_id = ol.customer_id WHERE c.name = 'Meridian Food'", _both(D),
         {"Meridian Food'": "Meridian Foods'"}),
    Case("C19", "control", "What is the largest order of sea salt flakes?",
         "SELECT max(ol.quantity) AS largest FROM order_line ol JOIN product p "
         "ON p.product_id = ol.product_id WHERE p.name = 'Sea-salt flakes'", _both(D),
         {"Sea-salt flakes": "Sea salt flakes"}),
    Case("C20", "control", "Which orders did Harrow Provisions place?",
         "SELECT ol.order_id FROM order_line ol JOIN customer c ON c.customer_id = ol.customer_id "
         "WHERE c.name = 'harrow provisions'", _both(D), {"harrow provisions": "Harrow Provisions"}),
]

_TYPED_SCHEMA = seed_demo.SCHEMA.replace("signup_date   TEXT", "signup_date   DATE").replace(
    "order_date    TEXT", "order_date    DATE")
_CARD = "TABLES customer, segment_lookup, product, order_line (the quickstart's demo schema)"
_GRANTS = GrantSet(frozenset({"customer", "segment_lookup", "product", "order_line"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def _rows():
    return (("customer", seed_demo.CUSTOMERS), ("segment_lookup", seed_demo.SEGMENTS),
            ("product", seed_demo.PRODUCTS), ("order_line", seed_demo.ORDER_LINES))


def _demo_file(path: str) -> DuckDBAdapter:
    """The quickstart's SQLite file, attached by DuckDB as `adapter_for` attaches a sqlite source."""
    con = sqlite3.connect(path)
    con.executescript(seed_demo.SCHEMA)
    for table, rows in _rows():
        con.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)
    con.commit()
    con.close()
    return DuckDBAdapter.sqlite(path)


def _typed_file(path: str) -> DuckDBAdapter:
    con = duckdb.connect(path)
    for statement in _TYPED_SCHEMA.split(";"):
        if statement.strip():
            con.execute(statement)
    for table, rows in _rows():
        con.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)
    con.close()
    return DuckDBAdapter.duckdb(path)


def _empty(rows: list) -> bool:
    """What a true "none" returns: no rows, one NULL row of aggregates, or a count of 0."""
    if not rows:
        return True
    return len(rows) == 1 and (all(v is None for v in rows[0]) or tuple(rows[0]) == (0,))


def run_question_set(workdir) -> dict[str, dict[str, str]]:
    """Each case's outcome per engine, after checking its premise against the data."""
    outcomes: dict[str, dict[str, str]] = {}
    for engine in ENGINES:
        build = _demo_file if engine == "demo" else _typed_file
        adapter = build(str(workdir / f"{engine}.{'sqlite' if engine == 'demo' else 'duckdb'}"))
        snap = enrich_structural(adapter, "demo")
        store = init_store(str(workdir / f"{engine}-store.duckdb"))
        build_value_index(adapter, snap, store)
        values = ValueIndex(store)
        verifiers, _ = _build_mode_verifiers(Settings(), client=object())
        for case in CASES:
            # The premise first: a "none" is empty on this data; a control's intended spelling
            # has rows and its misspelling has none, so "none" would be wrong.
            assert _empty(adapter.execute(case.sql)), (engine, case.id, "not a none on this data")
            if case.kind == "control":
                fixed = case.sql
                for wrong, right in case.fix.items():
                    assert fixed.count(wrong) == 1, (case.id, wrong)
                    fixed = fixed.replace(wrong, right)
                assert not _empty(adapter.execute(fixed)), (engine, case.id, fixed)
            agent = build_agent(MODES["instant"],
                                generator=FakeGenerator([json.dumps({"sql": case.sql})] * 6),
                                synthesizer=FakeSynthesizer("SYNTHESIZED"), adapter=adapter,
                                cache=TwoTierCache(L1Cache()), corrector=None, values=values,
                                selector=None, verifier=verifiers["instant"])
            packet = ContextPacket(question=case.question,
                                   cards=[RetrievedCard(object_id="customer", card=_CARD,
                                                        score=1.0)],
                                   grant_fingerprint=_GRANTS.fingerprint, enrichment_version="v1")
            ans = agent.answer(packet, snap, _GRANTS, _IDENTITY)
            assert not ans.failed, (engine, case.id, ans.answer)
            outcomes.setdefault(case.id, {})[engine] = D if ans.deferred else A
    return outcomes


def summary(outcomes: dict[str, dict[str, str]]) -> str:
    lines = []
    for engine in ENGINES:
        for kind, label in (("none", "genuine nones answered"), ("control", "controls declined")):
            ids = [c.id for c in CASES if c.kind == kind]
            want = A if kind == "none" else D
            got = sum(outcomes[i][engine] == want for i in ids)
            lines.append(f"{engine}: {label}: {got}/{len(ids)}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    found = run_question_set(tmp_path_factory.mktemp("genuine-none"))
    print("\n" + summary(found))
    return found


def test_the_set_is_twenty_and_twenty():
    assert sum(c.kind == "none" for c in CASES) == 20
    assert sum(c.kind == "control" for c in CASES) == 20
    assert len({c.id for c in CASES}) == 40
    assert all((c.fix is not None) == (c.kind == "control") for c in CASES)
    assert _TYPED_SCHEMA.count(" DATE ") == 2  # the typed engine's dates really are DATE


def test_every_misspelled_control_stays_declined(outcomes):
    wrong = {(c.id, e) for c in CASES if c.kind == "control" for e in ENGINES
             if outcomes[c.id][e] != D}
    assert wrong == set(), "a wrong 'none' went out"


@pytest.mark.parametrize("case", [c for c in CASES if c.kind == "none"], ids=lambda c: c.id)
def test_each_genuine_none_meets_its_preregistered_outcome(outcomes, case):
    assert outcomes[case.id] == case.expect
