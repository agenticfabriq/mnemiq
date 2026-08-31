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


def _payment_date_policy() -> Definition:
    """A policy definition: it rules on a table, and it matters most when the asker does not know
    to ask for it. Nobody types "payment date" when they ask "how many payments last quarter"."""
    return Definition(
        id="fspay:policy:payment_date",
        term="payment date",
        domain="fs_payments",
        definition="The business date is payment_transaction_date_time, not the other five.",
        bound_objects=["payment_transaction"],
    )


def test_a_bound_definition_rides_with_its_table():
    """Term-matching is right for a standard looked up by name and wrong for a policy that governs
    a table. Measured on the fs corpus: three of five policy definitions match no realistic
    question, so under term-matching alone they could never be retrieved however true they are."""
    grants = GrantSet(objects=frozenset({"payment_transaction"}))

    selected = select_definitions(
        "how many payments last quarter", [_payment_date_policy()], grants,
        table_ids=["payment_transaction"],
    )

    assert [d.term for d in selected] == ["payment date"]


def test_a_bound_definition_stays_with_its_table():
    # Non-vacuity: riding with a table must mean THAT table, not every question.
    grants = GrantSet(objects=frozenset({"payment_transaction", "refund"}))

    selected = select_definitions(
        "how many refunds", [_payment_date_policy()], grants, table_ids=["refund"]
    )

    assert selected == []


def test_an_unbound_standard_still_needs_its_term():
    """A public standard belongs to no table, so it has no table to ride with. It keeps the
    term-match rule, which is the right one for something looked up by name."""
    grants = GrantSet(objects=frozenset({"patient"}))

    assert select_definitions("what is an MPAA rating", [_standard()], grants, table_ids=["patient"])
    assert select_definitions("how many patients", [_standard()], grants, table_ids=["patient"]) == []


def test_a_definition_with_no_name_at_all_matches_no_question():
    """It used to match every one. The pattern for an empty term collapsed to a bare `\\b`, which
    is true at the start of any word, so such a definition was offered on every packet regardless
    of what was asked. `Definition.term` carries no non-empty constraint, so the shape is
    constructible.

    A definition with NO name -- no term and no legible id tail -- is retrievable only by riding
    with a table it is bound to. One that has a term-less id is named by that tail; see
    `test_a_definition_named_only_by_its_id_is_still_retrievable`.
    """
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract.semantic import Definition
    from mnemiq.semantic.glossary import select_definitions

    nameless = Definition(id="", term="", domain="fspay",
                          definition="incurred losses over earned premium",
                          bound_objects=["fs.payments"])
    grants = GrantSet(objects=frozenset({"fs.payments"}))
    assert select_definitions("loss ratio, revenue, anything", [nameless], grants) == []
    assert select_definitions("anything at all", [nameless], grants, ["fs.payments"]) == [nameless], (
        "it still rides with the table it is bound to"
    )


def test_a_definition_named_only_by_its_id_is_still_retrievable():
    """`generate.undefined_terms._spoken` already grounds a declared term against the tail of a
    definition's `id`, because a record may carry its name there rather than in `term`. Retrieval
    matched on `term` alone, so such a definition was never put in the packet -- and the guard,
    finding nothing to ground against, deferred an answerable question. One corpus, two rules,
    pointing the same way as the tolerance-width split did."""
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract.semantic import Definition
    from mnemiq.generate.undefined_terms import ungrounded_terms
    from mnemiq.semantic.glossary import select_definitions

    by_id = Definition(id="fspay:policy:loss_ratio", term="", domain="fspay",
                       definition="incurred losses over earned premium",
                       bound_objects=["fs.payments"])
    grants = GrantSet(objects=frozenset({"fs.payments"}))
    retrieved = select_definitions("what is our loss ratio", [by_id], grants)
    assert retrieved == [by_id], "the id's tail is a name a question can use"
    assert ungrounded_terms(["loss ratio"], retrieved) == [], (
        "and what retrieval found, the guard must ground"
    )
    assert select_definitions("what is our revenue", [by_id], grants) == []
