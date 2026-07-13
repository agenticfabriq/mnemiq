import pytest

from mnemiq.enrichment.proposals import PII_LEVELS, SEMANTIC_TYPES, parse_annotation

ALLOWED = {"fireplace": {"yes", "no"}, "claim_identifier": set()}


def _parse(raw):
    return parse_annotation(raw, "fireclaim", ALLOWED)


def test_parses_a_good_reply():
    ann = _parse(
        """{"columns": [
            {"name": "fireplace", "description": "Whether the property has a fireplace.",
             "semantic_type": "boolean", "pii_level": "none",
             "code_meanings": {"yes": "has a fireplace", "no": "has no fireplace"}}
        ]}"""
    )
    col = ann.columns[0]
    assert col.name == "fireplace"
    assert col.semantic_type == "boolean"
    assert col.code_meanings == {"yes": "has a fireplace", "no": "has no fireplace"}


def test_hallucinated_column_is_dropped():
    ann = _parse('{"columns": [{"name": "policy_holder_ssn", "pii_level": "pii"}]}')
    assert ann.columns == []  # we never asked about this column; it does not exist


def test_invented_code_is_dropped():
    ann = _parse(
        '{"columns": [{"name": "fireplace", "code_meanings": '
        '{"yes": "has one", "maybe": "invented"}}]}'
    )
    assert ann.columns[0].code_meanings == {"yes": "has one"}


def test_out_of_vocabulary_values_are_dropped():
    ann = _parse(
        '{"columns": [{"name": "fireplace", "semantic_type": "flammability", '
        '"pii_level": "extremely_secret"}]}'
    )
    col = ann.columns[0]
    assert col.semantic_type is None
    assert col.pii_level is None


def test_fenced_and_chatty_replies_still_parse():
    ann = _parse(
        'Sure! Here is the JSON:\n```json\n{"columns": '
        '[{"name": "fireplace", "semantic_type": "boolean"}]}\n```\nHope that helps!'
    )
    assert ann.columns[0].semantic_type == "boolean"


@pytest.mark.parametrize("raw", ["", "no.", "{", '{"columns": "not-a-list"}', "null"])
def test_garbage_never_raises(raw):
    assert _parse(raw).columns == []


def test_null_meanings_and_descriptions_are_dropped_not_stringified():
    ann = _parse(
        '{"columns": [{"name": "fireplace", "description": null, "code_meanings": {"yes": null}}]}'
    )
    col = ann.columns[0]
    assert col.description is None
    assert col.code_meanings == {}  # "I don't know" must not become the string "None"


def test_duplicate_columns_take_the_first():
    ann = _parse(
        '{"columns": [{"name": "fireplace", "semantic_type": "boolean"}, '
        '{"name": "fireplace", "semantic_type": "other"}]}'
    )
    assert len(ann.columns) == 1
    assert ann.columns[0].semantic_type == "boolean"


def test_vocabularies_are_closed():
    assert "boolean" in SEMANTIC_TYPES and "identifier" in SEMANTIC_TYPES
    assert set(PII_LEVELS) == {"none", "pii", "phi"}
