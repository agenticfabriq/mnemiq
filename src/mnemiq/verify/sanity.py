from __future__ import annotations

import math

import pyarrow as pa

from mnemiq.sql.values_check import ABSENT, FilterValue, ValueGrounding
from mnemiq.verify.verdict import VerifyVerdict

# Named in a decline, at most this many: the sentence is for a caller to act on, not an inventory.
_NAMED = 3
# The answered note's filters, at most this many characters of them.
_NOTE_FILTERS = 300

_UNSURE = "so I can't tell a true \"none\" from a misspelled value."


def _degenerate(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _listed(texts: list[str]) -> str:
    quoted = [repr(t) for t in texts]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _named(values: list[FilterValue]) -> str:
    """Values an answer could not rest on, one reason and one column between them, in the
    caller's words."""
    first, many = values[0], len(values) > 1
    if first.why == "not_a_plain_string":  # its text is the constant as written, quotes and all
        written = ", ".join(v.text for v in values)
        return f"{written}, a constant written other than as a plain string, could not be checked"
    if first.why == "json_path":  # the whole path as written
        written = ", ".join(v.text for v in values)
        return f"{written}, {'JSON paths' if many else 'a JSON path'}, could not be checked"
    texts, column = _listed([v.text for v in values]), first.column
    if first.outcome == ABSENT:
        return f"{texts} {'do' if many else 'does'} not appear among {column}'s known values"
    if first.why == "pattern":
        on = f" on {column}" if column else ""
        return f"the pattern{'s' if many else ''} {texts}{on} could not be checked against known values"
    if first.why == "date_on_time":
        return (f"the date{'s' if many else ''} {texts} against {column}, which holds times as well "
                "as dates, could not be checked as a bound")
    if first.why == "not_on_a_date_column":
        return (f"the date{'s' if many else ''} {texts}, not compared with a column known to hold "
                "dates alone, could not be checked as a bound")
    if first.why == "not_on_a_timestamp_column":
        return (f"the timestamp{'s' if many else ''} {texts}, not compared with a column known to "
                "hold timestamps, could not be checked as a bound")
    if first.why == "timestamp_equality":
        return (f"the timestamp{'s' if many else ''} {texts} compared for equality with {column} "
                "could not be checked as a bound")
    if first.why == "zone_mismatch":
        return (f"the timestamp{'s' if many else ''} {texts}"
                + (f" against {column}" if column else "")
                + ", one zone-aware and the other not, could not be checked as a bound")
    if first.why == "not_a_date":
        return f"{texts}, not a well-formed date, could not be checked"
    if first.why == "not_a_timestamp":
        return f"{texts}, not a well-formed timestamp with a time of day, could not be checked"
    if first.why == "typed_other":
        return (f"{texts}, typed as something other than a date or a timestamp, could not be "
                "checked as a bound")
    if first.why == "interval":
        return (f"the interval{'s' if many else ''} {texts}, not compared with a date or time "
                "column, could not be checked as a bound")
    if first.why == "range":
        return (f"the text bound{'s' if many else ''} {texts} on {column} could not be checked "
                "against known values")
    if first.why == "not_a_column":
        return f"{texts}, compared with an expression rather than a column, could not be checked"
    if first.why == "not_compared":
        return f"{texts}, compared with no column, could not be checked"
    if first.why == "unresolved" or column is None:
        return f"{texts}, on a column the check could not place, could not be checked"
    # not_indexed, masked: the same words, since a masked column's values are not this caller's
    # to learn about, and "masked" beside a value would say the check was possible.
    return f"{texts} could not be checked against {column}'s known values"


def _grounded_or_why(opening: str, grounding: ValueGrounding) -> str | None:
    """None when the answer may rest on what it filters on, else the sentence saying why not."""
    if grounding.never_true is not None:
        return f"{opening}, because the query's filter can never be true ({grounding.never_true})."
    blocking = grounding.blocking()
    if not blocking:
        return None
    groups: dict[tuple, list[FilterValue]] = {}
    for value in blocking:  # in the order written, one clause per reason and column
        groups.setdefault((value.outcome == ABSENT, value.why, value.column), []).append(value)
    named = [_named(group) for group in groups.values()]
    more = f"; and {len(named) - _NAMED} more" if len(named) > _NAMED else ""
    return f"{opening}, and {'; '.join(named[:_NAMED])}{more}, {_UNSURE}"


def _aggregate_over_nothing(table: pa.Table, rows: list[dict]) -> bool:
    """One row, every cell NULL: what SUM, AVG, MIN and MAX return over no rows."""
    return table.num_rows == 1 and all(v is None for v in rows[0].values())


def sanity_check(question: str, table: pa.Table,
                 grounding: ValueGrounding | None = None) -> VerifyVerdict | None:
    """Defer results that cannot be a real answer: no rows, a null/NaN scalar, or all-null cells.

    Except where the result is the answer to a question whose true answer is "none" (M146). An
    empty result, or one NULL row from aggregates over nothing, passes when `grounding` -- the
    decider's record of every string value in the statement, the projection's included -- says
    each was found in its column's indexed values or is a bound of a bare column's own type (a
    date on a DATE column or as the half-open end of a range on a TIMESTAMP one, a full timestamp on a
    TIMESTAMP one), and no predicate can never hold.
    A value on a table this identity reads through a row filter is never grounded: the index
    cannot say which rows the caller may read (`ground_values`). Anything
    nothing could check keeps it declined, and the decline names the value, so the caller can
    confirm or correct it. `grounding` None means it was not evaluated (an `Approved` built
    outside the decider) and the rules are as before, every such result declined.

    The rule it replaces was a proxy: an empty or null result presented as an answer was the
    single biggest catchable wrong-class in a 14B error analysis (~17% of wrong answers), mostly a
    wrong filter value. The value index is the evidence for that, held already. A wrong join or a
    wrong column over values that are all right is the residual an answered empty result carries,
    in `instant` where no judge reads it.
    """
    if table.num_rows == 0:
        if grounding is None:
            return VerifyVerdict(0.0, True, "The query returned no rows, so I can't answer this confidently.", "sanity")
        why = _grounded_or_why("No rows matched", grounding)
        return None if why is None else VerifyVerdict(0.0, True, why, "sanity")
    rows = table.to_pylist()
    if grounding is not None and _aggregate_over_nothing(table, rows):
        if grounding.null_as_written is not None:
            return VerifyVerdict(0.0, True, "The query produced an empty value, because it selects "
                                 f"NULL as written ({grounding.null_as_written}), not a value from "
                                 "the data.", "sanity")
        why = _grounded_or_why("The query produced an empty value", grounding)
        return None if why is None else VerifyVerdict(0.0, True, why, "sanity")
    if table.num_rows == 1 and table.num_columns == 1:
        if _degenerate(next(iter(rows[0].values()))):
            return VerifyVerdict(0.0, True, "The query produced an empty value, so I can't answer this confidently.", "sanity")
    if all(all(v is None for v in r.values()) for r in rows):
        return VerifyVerdict(0.0, True, "The query returned only empty values, so I can't answer this confidently.", "sanity")
    return None


def answered_note(table: pa.Table, grounding: ValueGrounding | None) -> str:
    """The engine's sentence on an answered empty result, appended after the synthesizer and never
    through it, like the other disclosures: a "none" must say what it is a "none" of. The filters
    are the statement as asked, never a row policy's predicate, cut at `_NOTE_FILTERS` characters
    where a long statement would bury the answer (the caller is shown the SQL whole). Empty for
    any other result."""
    filters = "; ".join(grounding.filters) if grounding is not None else ""
    if len(filters) > _NOTE_FILTERS:
        filters = (f"{filters[:_NOTE_FILTERS]}... (cut at {_NOTE_FILTERS} characters; the SQL "
                   "shown with this answer has them whole)")
    if table.num_rows == 0:
        return f"No rows matched: {filters}." if filters else "No rows matched."
    if _aggregate_over_nothing(table, table.to_pylist()):
        if grounding is not None and grounding.null_as_written is not None:
            return ""  # the query's own NULL, answered only with the layer off: no "none" to describe
        matched = f"no rows matched ({filters})" if filters else "no rows matched"
        return f"The result holds no value: {matched}, or every row that matched holds none."
    return ""
