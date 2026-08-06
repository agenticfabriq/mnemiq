"""History is bounded by the authorization boundary, then by size -- in that order.

The security property is the one worth stating twice: a turn answered under one policy
is never replayed to another, because its values would steer the next query and the
answer would disclose through a filter what the policy withheld from the column.
"""

from mnemiq.agent.history import MAX_ROWS, MAX_TURNS, scope_history
from mnemiq.contract import HistoryTurn


def _turn(question: str, fingerprint: str = "fp-analyst", rows: int = 1) -> HistoryTurn:
    return HistoryTurn(
        question=question,
        sql="SELECT region FROM policy",
        tables_used=["policy"],
        columns=["region", "total"],
        rows=[["west", i] for i in range(rows)],
        grant_fingerprint=fingerprint,
    )


def test_nothing_to_remember_is_not_an_error():
    assert scope_history(None, "fp-analyst") == []
    assert scope_history([], "fp-analyst") == []


def test_turns_from_this_boundary_are_kept():
    kept = scope_history([_turn("q1"), _turn("q2")], "fp-analyst")
    assert [t.question for t in kept] == ["q1", "q2"]


def test_a_turn_from_another_policy_is_dropped_entirely():
    # claims_lead saw policyholder names; analyst may not. Replaying that turn would let
    # a withheld value steer this query.
    history = [_turn("who are the policyholders?", fingerprint="fp-claims-lead"),
               _turn("how many claims?", fingerprint="fp-analyst")]

    kept = scope_history(history, "fp-analyst")

    assert [t.question for t in kept] == ["how many claims?"]


def test_a_dropped_turn_loses_its_question_and_sql_too_not_just_its_rows():
    # Half a turn is not safer than none: the question and the SQL were themselves shaped
    # by data this identity cannot see.
    kept = scope_history([_turn("privileged", fingerprint="fp-other")], "fp-analyst")

    assert kept == []


def test_an_empty_fingerprint_does_not_match_a_real_one():
    # A client that omits the field must not thereby inherit the current identity's scope.
    assert scope_history([_turn("q", fingerprint="")], "fp-analyst") == []


def test_only_the_last_few_turns_survive():
    history = [_turn(f"q{i}") for i in range(MAX_TURNS + 3)]

    kept = scope_history(history, "fp-analyst")

    assert len(kept) == MAX_TURNS
    assert kept[-1].question == f"q{MAX_TURNS + 2}", "the newest turn is the one nearest"


def test_rows_are_bounded_so_history_cannot_crowd_out_the_schema():
    kept = scope_history([_turn("q", rows=50)], "fp-analyst")

    assert len(kept[0].rows) == MAX_ROWS


def test_bounding_does_not_mutate_the_caller_s_turns():
    original = _turn("q", rows=50)

    scope_history([original], "fp-analyst")

    assert len(original.rows) == 50


def test_ordering_is_preserved_so_the_newest_turn_reads_last():
    kept = scope_history([_turn("older"), _turn("newer")], "fp-analyst")
    assert [t.question for t in kept] == ["older", "newer"]
