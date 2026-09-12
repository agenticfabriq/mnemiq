"""`strip_sample_values`'s clause regex, which is the whole of what that script decides.

Here rather than in a comment because the comment claiming this verification was the finding:
it asserted "identical substitutions" for work done in a scratch file that no one could re-run.
`tests/test_ddl_translate.py` is the convention -- a script's pure part gets a test.

The script defers its `snowflake.connector` import into `main()` so this module imports in CI,
where the warehouse extra is not installed.
"""

import time

from scripts.strip_sample_values import _SAMPLES


def strip(text: str) -> str:
    return _SAMPLES.sub("", text)


# The pattern's first character is the space before `sample_values`, so it is consumed with the
# clause -- `"  sample_values (...)"` leaves one space, not two. Asserted rather than trimmed,
# because that space is the join between the clause and whatever preceded it in the DDL.


def test_the_clause_is_removed_and_its_surroundings_are_not():
    assert strip("  a sample_values ('x', 'y') b") == "  a b"
    assert strip("  sample_values ('plain')") == " "
    assert strip("  sample_values ()") == " "
    assert strip("  no clause here") == "  no clause here"


def test_a_value_may_contain_the_characters_that_would_end_the_clause():
    """The reason this is not `\\(.*?\\)`. A doubled quote is SQL's escape, and a comma or a
    closing paren inside a value must not terminate anything."""
    assert strip("  sample_values ('with '' an escaped quote')") == " "
    assert strip("  sample_values ('has, comma', 'paren ) inside')") == " "
    assert strip("  sample_values ('multi'', ''escape', 'two')") == " "


def test_whitespace_around_the_separators_is_stripped_too():
    """The WIDENING the unrolled form introduced, asserted so it is a decision and not a
    surprise. Only the first two are new -- `('a' )` and `('a' , 'b')`, where whitespace sits
    against the separator -- and both are valid clauses. The last two matched before as well
    and are here so the widening cannot quietly become a narrowing."""
    assert strip("  sample_values ('a' )") == " "
    assert strip("  sample_values ('a' , 'b')") == " "
    assert strip("  sample_values ('a',   'b')") == " "
    assert strip("  sample_values ('trailing',)") == " "


# Each probe is sized so the pattern it guards against blows PAST the 0.5s bound while the
# current one finishes in under a millisecond -- measured, because a threshold sitting near
# the failing case stops catching it on a faster machine. The first draft of the space test
# used 6,400 spaces, where the quadratic form takes 148 ms and would have sailed through.


def test_two_clauses_in_one_statement_are_removed_separately():
    """Every other case here is one clause alone, which cannot catch the pattern running
    greedily from the first `sample_values (` to the LAST `)` and eating the columns between
    them. A real `GET_DDL` has one clause per column, so the two-clause case IS the normal
    case and the isolated ones are the artificial ones."""
    assert strip("a sample_values ('x') KEEP sample_values ('y') b") == "a KEEP b"
    # The paren inside a value is what makes this more than a repetition of the above: a
    # pattern that stopped at the first `)` would leave `paren') MID sample_values ('z'`.
    assert strip("c1 sample_values ('has ) paren') MID sample_values ('z') c2") == "c1 MID c2"


def test_the_pattern_is_linear_in_a_run_of_quotes():
    """py/redos. The form this replaced quadrupled every four quotes -- 0.01 ms at 14, 0.56 at
    26, 814 at 46 -- so 60 is minutes. The current form does 60 in 0.022 ms."""
    probe = " sample_values ('" + "'" * 60
    start = time.perf_counter()
    _SAMPLES.search(probe)
    assert (time.perf_counter() - start) < 0.5, "exponential backtracking is back"


def test_the_pattern_is_linear_in_a_run_of_SPACES():
    """The second split point, and the one the first fix introduced: `\\s*,?\\s*` is two
    unbounded `\\s*` either side of an optional comma. Quadratic rather than exponential, so it
    would never have blocked a merge: 3,068 ms here against 0.5 ms for `(?:\\s*,)?\\s*`, which
    binds the whitespace to the comma and leaves nothing to split."""
    probe = " sample_values ('a'" + " " * 25600 + "x"
    start = time.perf_counter()
    _SAMPLES.search(probe)
    assert (time.perf_counter() - start) < 0.5, "the whitespace split point is back"
