import json
import os

import pytest

from mnemiq.eval.bird import bird_db_path, load_bird

_REAL = "/Users/user/src/dataset/bird-minidev/MINIDEV"


@pytest.fixture
def mini(tmp_path):
    records = [
        {"question_id": 1, "db_id": "shop", "question": "how many?", "evidence": "count means COUNT(*)", "SQL": "SELECT count(*) FROM t", "difficulty": "simple"},
        {"question_id": 2, "db_id": "shop", "question": "avg?", "evidence": "", "SQL": "SELECT avg(x) FROM t", "difficulty": "moderate"},
        {"question_id": 3, "db_id": "bank", "question": "sum?", "evidence": "", "SQL": "SELECT sum(x) FROM t", "difficulty": "challenging"},
    ]
    path = tmp_path / "mini_dev_sqlite.json"
    path.write_text(json.dumps(records))
    return str(tmp_path)


def test_loads_every_case_with_routing_and_difficulty(mini):
    cases = load_bird(mini)
    assert [c.id for c in cases] == ["bird-1", "bird-2", "bird-3"]
    assert cases[0].db_id == "shop" and cases[2].db_id == "bank"
    assert cases[0].gold_sql == "SELECT count(*) FROM t"
    assert "simple" in cases[0].tags
    assert cases[0].answerable is True


def test_evidence_is_appended_by_default_and_omittable(mini):
    with_ev = load_bird(mini)[0].question
    without = load_bird(mini, with_evidence=False)[0].question
    assert "count means COUNT(*)" in with_ev
    assert "count means COUNT(*)" not in without
    assert without == "how many?"


def test_filters_compose(mini):
    assert [c.id for c in load_bird(mini, db_ids=["shop"])] == ["bird-1", "bird-2"]
    assert [c.id for c in load_bird(mini, difficulty="challenging")] == ["bird-3"]
    assert [c.id for c in load_bird(mini, limit=2)] == ["bird-1", "bird-2"]


def test_db_path_is_the_sqlite_file(mini):
    assert bird_db_path(mini, "shop").endswith("dev_databases/shop/shop.sqlite")


@pytest.mark.skipif(not os.path.isdir(_REAL), reason="real BIRD mini-dev not present")
def test_the_real_corpus_is_the_promised_size():
    cases = load_bird(_REAL)
    assert len(cases) == 500
    assert len({c.db_id for c in cases}) == 11
    assert all(os.path.isfile(bird_db_path(_REAL, c.db_id)) for c in cases)
