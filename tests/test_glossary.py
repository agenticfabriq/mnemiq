import json

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Definition
from mnemiq.semantic.glossary import load_definitions, select_definitions


def _premium() -> Definition:
    return Definition(
        id="def-premium",
        term="premium",
        domain="policy",
        definition="Premium amounts live in policy_amount, joined through premium.",
        bound_objects=["premium", "policy_amount"],
    )


def _standard() -> Definition:
    # A public standard (e.g. MPAA/ICD-10): its text names no table, so it is safe for anyone.
    return Definition(
        id="ontology:scheme:mpaa", term="MPAA rating", domain="ontology",
        definition="Motion Picture Association film-rating scheme.", public=True,
    )


def _internal() -> Definition:
    # A confidential internal taxonomy shipped unbound: must NOT be globally visible.
    return Definition(
        id="ontology:scheme:layoff", term="layoff tier", domain="ontology",
        definition="Internal workforce-reduction tiering.",
    )


def _grants(*objects: str) -> GrantSet:
    return GrantSet(frozenset(objects))


def test_a_public_definition_is_visible_without_any_grant():
    # Public standards keep the old unbound behavior -- but now it is explicit, not incidental.
    assert select_definitions("what is the MPAA rating?", [_standard()], _grants()) == [_standard()]


def test_an_unbound_nonpublic_definition_is_hidden_from_everyone():
    # The SP1 leak: an unbound def was shown to all (all([]) is True). Fail closed now: a
    # confidential taxonomy that no one was granted is visible to no one.
    assert select_definitions("what layoff tier is this?", [_internal()], _grants("layoff")) == []


def test_load_definitions_round_trips(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps([_premium().model_dump()]))
    (loaded,) = load_definitions(str(path))
    assert loaded == _premium()


def test_a_matching_term_is_selected():
    got = select_definitions(
        "what is the total premium amount?", [_premium()], _grants("premium", "policy_amount")
    )
    assert got == [_premium()]


def test_matching_is_case_insensitive_and_tolerates_inflection():
    grants = _grants("premium", "policy_amount")
    assert select_definitions("Total PREMIUMS please", [_premium()], grants)


def test_a_substring_inside_a_word_does_not_match():
    grants = _grants("premium", "policy_amount")
    assert not select_definitions("is unpremium a word?", [_premium()], grants)


def test_an_unrelated_question_selects_nothing():
    assert not select_definitions(
        "how many claims are there?", [_premium()], _grants("premium", "policy_amount")
    )


def test_a_definition_binding_an_ungranted_table_is_never_shown():
    # showing it would leak that policy_amount exists AND steer the model into a
    # table the decider must then reject -- fail closed
    got = select_definitions("total premium?", [_premium()], _grants("premium"))
    assert got == []


def test_no_grants_no_definitions():
    assert select_definitions("total premium?", [_premium()], _grants()) == []


def test_a_multi_word_term_matches_as_a_phrase():
    loss = Definition(
        id="def-loss-ratio", term="loss ratio", domain="claims",
        definition="Losses over premiums.", bound_objects=["fireclaim"],
    )
    grants = _grants("fireclaim")
    assert select_definitions("average loss ratios by year", [loss], grants)
    assert not select_definitions("ratio of losses", [loss], grants)  # order matters
