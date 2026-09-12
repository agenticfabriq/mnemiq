"""`plan_recasts`, which decides which columns get rewritten to make a join's types agree.

Pure — it reads a type map and returns a plan — so these need no warehouse and no fake
cursor. It was inside `main` alongside live Snowflake I/O until this change, which is why the
defect below survived: nothing could reach the decision without a database.

The defect: the direction was decided ONE PAIR AT A TIME, with no regard for the other
children of the same parent column. Moving a parent to suit one child silently broke every
sibling that already agreed with it, and the script reported only what it aligned.
"""

from scripts.align_fk_types_snowflake import plan_recasts

NUM = "NUMBER(38,0)"


def merged(plan) -> dict:
    """The plan flattened to {table: {column: target}}.

    `plan_recasts` returns ONE ENTRY PER COMPONENT, because the caller has to treat a
    component as all-or-nothing -- probing per table and skipping one of them splits the
    component the grouping exists to hold together. Most assertions here are about which
    columns move, so they read the flattened view; the ones about component BOUNDARIES read
    the list itself.
    """
    out: dict = {}
    for component in plan:
        for table, columns in component.items():
            out.setdefault(table, {}).update(columns)
    return out


def test_moving_a_parent_takes_its_other_children_with_it():
    """THE REGRESSION. `P.K` TEXT with a TEXT child and a NUMBER child.

    Pair-at-a-time: C1/P agree, so that pair is skipped as fine; then C2 makes `P.K` numeric
    and C1 is left TEXT against a NUMBER parent -- its FOREIGN KEY no longer declarable, and
    `aligned 1 columns` printed with nothing said about it. Whichever pair came second decided
    the parent's type and the other one lost.

    The group moves together now, so both children can still declare their key.
    """
    plan = merged(plan_recasts(
        [("C1", "K", "P", "K"), ("C2", "K", "P", "K")],
        {("P", "K"): "TEXT", ("C1", "K"): "TEXT", ("C2", "K"): NUM},    ))
    assert plan == {"P": {"K": NUM}, "C1": {"K": NUM}}, plan


def test_two_disagreeing_parents_are_reconciled_rather_than_one_being_dropped():
    """`C.K` TEXT referenced by `P1.K NUMBER` and `P2.K FLOAT`.

    A column holds one type, so under the pair rule one of those relationships had to lose --
    and which one followed the input order. Under components all three are ONE component, so
    the disagreeing PARENT moves too and both keys stay declarable. Worth stating plainly
    because it means a child's type can now rewrite a parent table, which is a bigger
    consequence than the pair rule ever had.

    The target is `_numeric_rank`, not input order: an id is an identifier, so NUMBER outranks
    FLOAT. The VALUE is asserted and not just forward == reverse -- an earlier version checked
    only that the two orders agreed, which two identical wrong answers also satisfy.

    (That earlier version also used the shared-PARENT case and claimed the pair rule was
    order-dependent there. It was not: `types` is read fresh per pair and never updated, so
    both orders agreed and the test passed against the bug it named.)
    """
    types = {("C", "K"): "TEXT", ("P1", "K"): NUM, ("P2", "K"): "FLOAT"}
    forward = merged(plan_recasts([("C", "K", "P1", "K"), ("C", "K", "P2", "K")], types))
    reverse = merged(plan_recasts([("C", "K", "P2", "K"), ("C", "K", "P1", "K")], types))
    assert forward == reverse, f"the plan followed the input order: {forward} != {reverse}"
    assert forward == {"C": {"K": NUM}, "P2": {"K": NUM}}, forward


def test_the_widest_numeric_wins_rather_than_the_lexicographically_smallest():
    """A tie inside one numeric family is broken by declared precision and scale, not by the
    rendered string.

    A plain string compare put `NUMBER(10,0)` before `NUMBER(38,0)` -- "1" sorts before "3" --
    so a 38-digit parent was NARROWED to ten digits, while against `NUMBER(9,0)` the same
    compare widened instead: the direction flipped on the first character of the digit count.
    Widening cannot lose a value; narrowing can, and would surface only as the loss probe
    refusing the whole component.
    """
    def target_for(parent, child):
        return merged(plan_recasts([("C", "K", "P", "K")],
                                   {("P", "K"): parent, ("C", "K"): child}))

    assert target_for("NUMBER(38,0)", "NUMBER(10,0)") == {"C": {"K": "NUMBER(38,0)"}}
    assert target_for("NUMBER(10,0)", "NUMBER(38,0)") == {"P": {"K": "NUMBER(38,0)"}}
    # Scale widens too, and an integer family still outranks a floating one.
    assert target_for("NUMBER(10,2)", "NUMBER(10,4)") == {"P": {"K": "NUMBER(10,4)"}}
    assert target_for("NUMBER(38,0)", "FLOAT") == {"C": {"K": "NUMBER(38,0)"}}


def test_independent_components_are_separate_entries_so_one_can_be_skipped():
    """The list shape is load-bearing: the caller probes a component and skips ALL of it when
    any member is lossy. Flattening two independent components into one map would make an
    unrelated table's failure skip this one too."""
    plan = plan_recasts(
        [("C1", "K", "P1", "K"), ("C2", "K", "P2", "K")],
        {("P1", "K"): NUM, ("C1", "K"): "TEXT", ("P2", "K"): "FLOAT", ("C2", "K"): "TEXT"},
    )
    assert len(plan) == 2, plan
    assert {frozenset(c) for c in plan} == {frozenset({"C1"}), frozenset({"C2"})}


def test_a_chain_is_ONE_entry_so_it_cannot_be_half_applied():
    """The converse. G -> C -> P is one component, so it is one entry and the caller either
    rewrites all of it or none -- rewriting `C` while skipping `P` is the split that grouping
    exists to prevent, and a per-table plan could not express the difference."""
    plan = plan_recasts(
        [("C2", "K", "P", "K"), ("C", "K", "P", "K"), ("G", "K", "C", "K")],
        {("P", "K"): "TEXT", ("C2", "K"): NUM, ("C", "K"): "TEXT", ("G", "K"): "TEXT"},
    )
    assert len(plan) == 1, plan
    assert set(plan[0]) == {"P", "C", "G"}, plan[0]


def test_a_column_that_moves_as_a_child_carries_ITS_children_too():
    """The defect that grouping-by-parent introduced while fixing the pair rule, caught by
    review before it shipped.

    Chain: G1/G2/G3 -> C -> P, with a numeric C2 also pointing at P. Grouping by parent moved
    `C.K` to NUMBER as part of P's group, then read `C.K`'s ORIGINAL text type when deciding
    C's own group and left all three grandchildren TEXT against a now-NUMBER parent. Three
    relationships broken where the pair rule broke one -- and silently, since `main` prints
    only what it aligned.

    Foreign keys chain, so the unit is the connected COMPONENT: every column reachable
    through a key ends at the same type.
    """
    plan = merged(plan_recasts(
        [("C2", "K", "P", "K"), ("C", "K", "P", "K"),
         ("G1", "K", "C", "K"), ("G2", "K", "C", "K"), ("G3", "K", "C", "K")],
        {("P", "K"): "TEXT", ("C2", "K"): NUM, ("C", "K"): "TEXT",
         ("G1", "K"): "TEXT", ("G2", "K"): "TEXT", ("G3", "K"): "TEXT"},    ))
    assert plan == {t: {"K": NUM} for t in ("P", "C", "G1", "G2", "G3")}, plan


def test_a_component_holding_two_numeric_types_picks_one_by_RANK_not_input_order():
    """Two numeric children of one parent, and only the groups were sorted before -- the
    target came from whichever numeric child appeared first in `foreign`, so the same schema
    gave `{P: NUMBER, C2: NUMBER}` one way and `{P: FLOAT, C1: FLOAT}` the other.

    An id is an identifier, so NUMBER outranks FLOAT -- the same reason the loss probe rejects
    a fractional column. Whichever way the keys are listed, the component lands on NUMBER and
    every member agrees, which is what keeps all of its relationships declarable.
    """
    types = {("P", "K"): "TEXT", ("C1", "K"): NUM, ("C2", "K"): "FLOAT"}
    pairs = [("C1", "K", "P", "K"), ("C2", "K", "P", "K")]
    forward = merged(plan_recasts(pairs, types))
    reverse = merged(plan_recasts(list(reversed(pairs)), types))
    assert forward == reverse, f"the target followed input order: {forward} != {reverse}"
    assert forward == {"P": {"K": NUM}, "C2": {"K": NUM}}, forward


def test_a_numeric_parent_keeps_its_type_and_the_children_come_to_it():
    """An id is a number: the group never degrades to text when the parent already holds the
    numeric type. This is the branch that moved the child to the parent."""
    plan = merged(plan_recasts(
        [("C1", "K", "P", "K"), ("C2", "K", "P", "K")],
        {("P", "K"): NUM, ("C1", "K"): "TEXT", ("C2", "K"): "TEXT"},    ))
    assert plan == {"C1": {"K": NUM}, "C2": {"K": NUM}}


def test_a_group_that_already_agrees_is_not_rewritten():
    """Rewriting a table costs a CREATE OR REPLACE over live data. Agreement means no plan."""
    assert plan_recasts([("C1", "K", "P", "K")], {("P", "K"): NUM, ("C1", "K"): NUM}) == []


def test_an_all_text_group_is_left_alone():
    """No member is numeric, so there is nothing to prefer and nothing to gain: casting text
    to text rewrites two tables to change nothing."""
    assert plan_recasts([("C1", "K", "P", "K")], {("P", "K"): "TEXT", ("C1", "K"): "TEXT"}) == []


def test_a_column_missing_from_the_type_map_is_skipped_not_guessed():
    """`column_types` reports what the schema has. A pair naming something it does not is a
    stale tables.json entry, and inventing a type for it would rewrite the wrong column."""
    assert plan_recasts([("C1", "K", "P", "K")], {("P", "K"): NUM}) == []
    assert plan_recasts([("C1", "K", "P", "K")], {("C1", "K"): NUM}) == []


def test_two_independent_parents_do_not_reach_into_each_other():
    plan = merged(plan_recasts(
        [("C1", "K", "P1", "K"), ("C2", "K", "P2", "K")],
        {("P1", "K"): NUM, ("C1", "K"): "TEXT", ("P2", "K"): "TEXT", ("C2", "K"): "TEXT"},    ))
    assert plan == {"C1": {"K": NUM}}, plan


def test_one_table_can_collect_two_columns_in_a_single_rebuild():
    """`main` issues ONE `CREATE OR REPLACE` per table, with a projection covering every
    column in that table's entry -- so two moving columns of one child must arrive in a single
    entry. Two entries for one table would mean the second rebuild reads the first's output."""
    plan = merged(plan_recasts(
        [("C", "A", "P", "A"), ("C", "B", "P", "B")],
        {("P", "A"): NUM, ("C", "A"): "TEXT", ("P", "B"): NUM, ("C", "B"): "TEXT"},    ))
    assert plan == {"C": {"A": NUM, "B": NUM}}, plan
