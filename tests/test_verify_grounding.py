from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.verify.grounding import grounding_check


def _packet(q):
    return ContextPacket(question=q, cards=[], grant_fingerprint="", enrichment_version=None)


def _approved(sql):
    return Approved(plan_sql=sql, target_sql=sql)


def test_quoted_value_absent_from_sql_defers():
    p = _packet("average consumption of customers in 'SME'")
    a = _approved("SELECT AVG(Consumption) FROM yearmonth")
    v = grounding_check(p, a)
    assert v is not None and v.defer and v.layer == "grounding"


def test_year_absent_from_sql_defers():
    p = _packet("total sales for the year 2013")
    a = _approved("SELECT SUM(amount) FROM sales")
    assert grounding_check(p, a) is not None


def test_value_present_in_sql_passes():
    p = _packet("average consumption of customers in 'SME'")
    a = _approved("SELECT AVG(c.Consumption) FROM customers c WHERE c.Segment = 'SME'")
    assert grounding_check(p, a) is None


def test_question_without_salient_literals_passes():
    p = _packet("how many customers are there?")
    a = _approved("SELECT COUNT(*) FROM customers")
    assert grounding_check(p, a) is None


def test_case_insensitive_match_passes():
    # question says "discount"; SQL uses 'Discount' -> grounded
    p = _packet('how many "discount" stations?')
    a = _approved("SELECT COUNT(*) FROM gasstations WHERE Segment = 'Discount'")
    assert grounding_check(p, a) is None


def test_possessive_apostrophe_is_not_a_quoted_value():
    # "Sanders's" must not be parsed as a quoted literal 's'
    p = _packet("What's Angela Sanders's major?")
    a = _approved("SELECT major_name FROM member JOIN major WHERE first_name = 'Angela'")
    assert grounding_check(p, a) is None


def test_reformatted_date_is_not_flagged():
    p = _packet("what segment at '2012/8/23'?")
    a = _approved("SELECT Segment FROM customers WHERE Date = '2012-08-23'")
    assert grounding_check(p, a) is None
