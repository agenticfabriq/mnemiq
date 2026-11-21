"""The table-search check: does retrieval find the tables each question needs, and does the prompt fit the window?

At a few tables every table reaches the prompt. At hundreds, retrieval picks `retrieval_k` of them, and a question whose
tables it misses cannot be answered whatever the model does; and wide cards can outgrow the window while every table
is found. `check` asks each question as answering would, up to the model call (`Runtime.search`: the same retrieval and
the same fit, no model asked), takes the tables its gold SQL reads (the engine's own reading, `sql.scope.base_tables`),
and says where each was found and whether the prompt fits. A gold query that does not parse, or reads a table the
deployment does not hold, is reported apart and never counted as a search miss. `render` prints counts and question ids
only, unless asked for names, so a deployment can send its result without its schema.
"""
from __future__ import annotations

from typing import Any


def gold_tables(sql: str, dialect: str | None) -> set[str] | None:
    """The tables `sql` reads, by the engine's reading; None when it does not parse."""
    import sqlglot

    from mnemiq.sql.scope import base_tables

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 -- reported as unreadable gold, never a miss
        return None
    if tree is None:
        return None
    try:
        # As written: `orders`, `SHOP.ORDERS`, `erp.orders`. `check` matches each to the id the deployment holds.
        return {".".join(p for p in (t.catalog, t.db, t.name) if p) for t in base_tables(tree, dialect)}
    except Exception:  # noqa: BLE001 -- as above
        return None


def _held_id(written: str, held: dict[str, str]) -> str | None:
    """The deployment's id for a table as the gold SQL wrote it: the longest dotted suffix the snapshot holds (a
    federated `erp.orders` as itself, a schema-qualified `SHOP.ORDERS` as `ORDERS`), ignoring case."""
    parts = written.split(".")
    for n in range(len(parts)):
        found = held.get(".".join(parts[n:]).casefold())
        if found is not None:
            return found
    return None


def check(runtime: Any, questions: list[dict], identity: Any) -> tuple[list[dict], dict]:
    """One row per question (`id`, then `gold`, `found`, `ranks`, `fits`, `tokens`, `budget`, `tables`, or
    `unreadable`), and the totals."""
    held = {b.object_id.casefold(): b.object_id for b in runtime.snapshot.source_bindings}
    dialect = getattr(runtime.adapter, "dialect", None)
    rows: list[dict] = []
    for q in questions:
        row: dict = {"id": q["id"]}
        gold = gold_tables(q["gold_sql"], dialect)
        if gold is None:
            row["unreadable"] = "the gold SQL does not parse"
        elif not gold:
            row["unreadable"] = "the gold SQL reads no table"
        elif outside := [g for g in gold if _held_id(g, held) is None]:
            row["unreadable"] = f"the gold SQL reads {len(outside)} table(s) outside this deployment"
        if "unreadable" in row:
            rows.append(row)
            continue
        text = q["gold_sql"].casefold()
        ordered = sorted({_held_id(g, held) for g in gold}, key=lambda g: (text.find(g.casefold()), g.casefold()))
        result = runtime.search(q["question"], identity)
        ranked = [t.casefold() for t in result.tables]
        ranks = [ranked.index(g.casefold()) + 1 if g.casefold() in ranked else None for g in ordered]
        fit = result.fit
        row.update(gold=len(ordered), found=sum(r is not None for r in ranks), ranks=ranks, tables=ordered,
                   fits=getattr(fit, "fits", None), tokens=getattr(fit, "tokens", None),
                   budget=getattr(fit, "budget", None), retrieved=len(result.tables))
        rows.append(row)
    checked = [r for r in rows if "unreadable" not in r]
    summary = {"questions": len(rows), "checked": len(checked), "unreadable": len(rows) - len(checked),
               "all_found": sum(r["found"] == r["gold"] for r in checked),
               "gold_tables": sum(r["gold"] for r in checked), "found_tables": sum(r["found"] for r in checked),
               "fits": sum(r["fits"] is True for r in checked), "over_window": sum(r["fits"] is False for r in checked),
               "fit_unknown": sum(r["fits"] is None for r in checked)}
    return rows, summary


def render(rows: list[dict], summary: dict, names: bool = False) -> str:
    """The check as text: a line per question and the totals. No table name unless `names`."""
    lines = []
    for r in rows:
        if "unreadable" in r:
            lines.append(f"{r['id']}: not checked, {r['unreadable']}")
            continue
        fit = {True: "fits", False: "does NOT fit", None: "fit unknown"}[r["fits"]]
        sized = f" ({r['tokens']} of {r['budget']} tokens)" if r["budget"] is not None else ""
        line = f"{r['id']}: found {r['found']} of {r['gold']} gold table(s), ranks {r['ranks']} of {r['retrieved']}; prompt {fit}{sized}"
        if names:
            line += f"; gold tables {', '.join(r['tables'])}"
        lines.append(line)
    s = summary
    share = f"{100 * s['found_tables'] / s['gold_tables']:.0f}%" if s["gold_tables"] else "n/a"
    lines.append(f"{s['questions']} questions: {s['checked']} checked, {s['unreadable']} not checked (gold unreadable "
                 f"or outside the deployment); every gold table found for {s['all_found']} of {s['checked']}; "
                 f"{s['found_tables']} of {s['gold_tables']} gold tables found ({share}); prompt fits for {s['fits']}, "
                 f"over the window for {s['over_window']}, unknown for {s['fit_unknown']}")
    return "\n".join(lines)
