"""`splice`, which rewrites a semantic view's DDL and hands the result straight to
CREATE OR REPLACE against a live view.

It had no test because the module imported `snowflake.connector` at top level and so could
not be loaded without the warehouse extra. That import is deferred now, for the reason
`tests/test_ddl_translate.py` sets the convention: a script's pure part gets a test.

Every case below is a shape the function's own comments name -- a composite key sweeping in a
foreign key column, a relationship pointing at a column the parent cannot declare, and
Sakila's real STORE/STAFF cycle -- so a rewrite that stops handling one of them goes red
rather than producing a view Cortex Analyst quietly refuses.
"""

from scripts.add_relationships_spider2 import splice

# The two-dotted DIMENSION line must not collect a primary key; only the three-dotted table
# entries may. That distinction is a regex in `splice` and is the easiest thing to break.
DDL = (
    "create or replace semantic view SPIDER2.SAKILA.SV\n"
    "\ttables (\n"
    '\t\tSPIDER2.SAKILA."STORE" comment=\'a store\',\n'
    '\t\tSPIDER2.SAKILA."STAFF" primary key (MANAGER_STAFF_ID, STAFF_ID),\n'
    '\t\tSPIDER2.SAKILA."RENTAL"\n'
    "\t)\n"
    "\tfacts (\n"
    "\t\tSTORE.STORE_ID as store_id\n"
    "\t)\n"
    "\tdimensions (\n"
    "\t\tSTAFF.NAME as name\n"
    "\t)\n"
)


def test_a_view_that_already_has_relationships_is_left_alone():
    """Returns None rather than splicing a second block in. The caller skips on None, so this
    is what stops a re-run doubling the clause."""
    already = DDL.replace("\tfacts (", "\trelationships (\n\t\tX as A(B) references C(D)\n\t)\n\tfacts (")
    assert splice(already, {}, [("A", "B", "C", "D")]) is None


def test_no_declarable_relationship_yields_None_rather_than_an_empty_block():
    """An empty `relationships ()` is not the same as leaving the view unedited: it is a
    rewrite that changes nothing and costs a CREATE OR REPLACE against a live view."""
    assert splice(DDL, {}, []) is None


def test_the_primary_key_lands_on_table_entries_and_not_on_a_dimension():
    out = splice(DDL, {"STORE": "STORE_ID"}, [("RENTAL", "STORE_ID", "STORE", "STORE_ID")])
    assert out is not None
    assert 'SPIDER2.SAKILA."STORE" primary key (STORE_ID)' in out
    # The two-dotted lines are columns, not tables, and must be untouched.
    assert "\t\tSTORE.STORE_ID as store_id\n" in out
    assert "\t\tSTAFF.NAME as name\n" in out


def test_a_composite_key_is_narrowed_to_the_column_that_identifies_the_row():
    """Autopilot sweeps foreign key columns into the key -- SQLITE-SAKILA's STORE gets
    (MANAGER_STAFF_ID, STORE_ID). A relationship must name the whole key, so the composite is
    replaced rather than added to."""
    out = splice(DDL, {"STAFF": "STAFF_ID"}, [("STORE", "MANAGER_STAFF_ID", "STAFF", "STAFF_ID")])
    assert out is not None
    assert 'SPIDER2.SAKILA."STAFF" primary key (STAFF_ID)' in out
    assert "MANAGER_STAFF_ID, STAFF_ID" not in out, "the composite survived"


def test_only_one_referenced_column_survives_and_SORT_ORDER_picks_it():
    """A semantic view accepts a relationship only against the target's declared key, so where
    two columns of one parent are referenced, one relationship has to go.

    Which one is decided by `sorted(foreign)`, not by `primary`: the rule preferring "a key
    something actually references" fires on the first referenced column it sees and then
    locks, so EMAIL beats the declared STAFF_ID purely because it sorts first. That is the
    documented intent -- the key is "dictated by what points at it, not by whatever
    SHOW PRIMARY KEYS happens to report" -- but the tie-break between two referenced columns
    is alphabetical rather than principled, and this test exists to make that visible rather
    than to bless it. Reversing the two tuples below changes the surviving relationship.
    """
    out = splice(
        DDL,
        {"STAFF": "STAFF_ID"},
        [("STORE", "MANAGER_STAFF_ID", "STAFF", "STAFF_ID"),
         ("RENTAL", "STAFF_EMAIL", "STAFF", "EMAIL")],
    )
    assert out is not None
    assert "RENTAL_TO_STAFF as RENTAL(STAFF_EMAIL) references STAFF(EMAIL)" in out
    assert "STORE_TO_STAFF" not in out, "both survived; the view would be rejected"
    # And the table entry carries the key the surviving relationship needs, not the PK.
    assert 'SPIDER2.SAKILA."STAFF" primary key (EMAIL)' in out


def test_sakilas_real_cycle_keeps_one_edge_and_drops_the_one_that_closes_it():
    """STORE names its manager in STAFF and STAFF names the store it works at. Semantic views
    reject cycles, and a join path in one direction beats none in either."""
    out = splice(
        DDL,
        {"STAFF": "STAFF_ID", "STORE": "STORE_ID"},
        [("STORE", "MANAGER_STAFF_ID", "STAFF", "STAFF_ID"),
         ("STAFF", "STORE_ID", "STORE", "STORE_ID")],
    )
    assert out is not None
    assert ("STORE_TO_STAFF" in out) != ("STAFF_TO_STORE" in out), "kept both, or neither"


def test_two_relationships_are_comma_separated_and_the_repeat_name_is_suffixed():
    """Every other case here yields ONE relationship line, which leaves two things unreached:
    the `",\n".join` separator and the `while name in used` suffix loop. Mutating the join to
    `"\n".join` emits a block with no commas -- rejected by Snowflake for every real schema --
    and every other test stays green.

    Two foreign keys from one child to one parent key is the shape that reaches both at once:
    Sakila's RENTAL names a staff member twice. Both point at STAFF's declared key, so both
    survive the key rule, and the second collides on `RENTAL_TO_STAFF`.
    """
    out = splice(
        DDL,
        {"STAFF": "STAFF_ID"},
        [("RENTAL", "STAFF_ID", "STAFF", "STAFF_ID"),
         ("RENTAL", "RETURN_STAFF_ID", "STAFF", "STAFF_ID")],
    )
    assert out is not None
    assert "RENTAL_TO_STAFF as RENTAL(RETURN_STAFF_ID) references STAFF(STAFF_ID)" in out
    assert "RENTAL_TO_STAFF_2 as RENTAL(STAFF_ID) references STAFF(STAFF_ID)" in out
    # The separator. Without it the two lines run together and the DDL will not parse.
    block = out[out.index("\trelationships ("):out.index("\tfacts (")]
    assert block.count(",\n") == 1, f"entries are not comma-separated: {block!r}"


def test_the_block_is_placed_before_facts_where_snowflake_writes_it():
    out = splice(DDL, {"STORE": "STORE_ID"}, [("RENTAL", "STORE_ID", "STORE", "STORE_ID")])
    assert out is not None
    assert out.index("\trelationships (") < out.index("\tfacts ("), "clause is out of order"
    assert "RENTAL_TO_STORE as RENTAL(STORE_ID) references STORE(STORE_ID)" in out
