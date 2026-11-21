/**
 * What chart, if any, does this result want to be?
 *
 * The engine emits no chart spec, and we do not ask a model to invent one. The
 * result's own shape decides the form, deterministically: the data's job picks the
 * chart, and often the answer is "not a chart" -- a scalar is a number, and a
 * thousand categories is a table. Refusing to plot is a valid outcome here for the
 * same reason it is in the engine.
 */

import type { Cell, ResultPreview } from "./types";

export type ChartKind = "bar" | "line" | "scatter";

export type ChartSpec = {
  kind: ChartKind;
  /** Column supplying the x axis / category labels. */
  labelColumn: string;
  /** One entry per plotted series; more than one means a legend is required. */
  valueColumns: string[];
  /** Reason a chart is *not* offered, when spec is null. */
};

/** Above this, a categorical axis is unreadable and the table is the better view. */
const MAX_CATEGORIES = 40;

const isBlank = (v: Cell) => v === null || v === undefined || v === "";

/** Postgres numerics arrive as strings, so "1000.00" has to count as a number. */
export function isNumeric(values: Cell[]): boolean {
  const present = values.filter((v) => !isBlank(v));
  if (present.length === 0) return false;
  return present.every(
    (v) =>
      typeof v === "number" ||
      (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))),
  );
}

const DATE_LIKE = /^\d{4}-\d{2}(-\d{2})?([ T]|$)/;

export function isTemporal(values: Cell[]): boolean {
  const present = values.filter((v) => !isBlank(v));
  if (present.length === 0) return false;
  return present.every((v) => typeof v === "string" && DATE_LIKE.test(v.trim()));
}

export const toNumber = (v: Cell): number | null => {
  if (isBlank(v)) return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};

// An identifier word as the whole name or after `_` (`id`, `order_no`, `zip_code`), or a camel-case `Id` ending.
const IDENTIFIER = /(^|_)(id|key|code|no|number)$/i;
const CAMEL_ID = /[a-z0-9]Id$/;
const YEAR_NAME = /year|yr/i;

/**
 * Whether a column names its rows rather than measuring them: an identifier by its name, or a year, a column whose
 * name holds `year` or `yr` and whose every present value is a whole number from 1000 to 9999. A column whose values
 * merely fall in a year's range, with no year in its name, stays a measure.
 */
export function isLabelColumn(name: string, values: Cell[]): boolean {
  if (IDENTIFIER.test(name) || CAMEL_ID.test(name)) return true;
  if (!YEAR_NAME.test(name)) return false;
  const present = values.filter((v) => !isBlank(v));
  return (
    present.length > 0 &&
    present.every((v) => {
      const n = toNumber(v);
      return n !== null && Number.isInteger(n) && n >= 1000 && n <= 9999;
    })
  );
}

/**
 * Returns the chart this result should render as, or null when it should not be
 * charted at all.
 */
export function chartSpecFor(preview: ResultPreview): ChartSpec | null {
  const { columns, rows } = preview;
  if (columns.length < 2 || rows.length < 2) return null; // a scalar is a number
  if (rows.length > MAX_CATEGORIES) return null;

  const columnValues = columns.map((_, i) => rows.map((r) => (r[i] ?? null) as Cell));
  // A year or an identifier is numeric, but it names the rows: a label, never a measure.
  const numeric = columns.filter(
    (c, i) => isNumeric(columnValues[i]!) && !isLabelColumn(c, columnValues[i]!),
  );
  const temporal = columns.filter((_, i) => isTemporal(columnValues[i]!));

  if (numeric.length === 0) return null;

  // Change over time reads as a line, whatever else is in the row.
  const timeColumn = temporal.find((c) => !numeric.includes(c));
  if (timeColumn) {
    const values = numeric.filter((c) => c !== timeColumn);
    if (values.length > 0) return { kind: "line", labelColumn: timeColumn, valueColumns: values };
  }

  const labels = columns.filter((c) => !numeric.includes(c));

  // Identity vs magnitude: the classic group-by result.
  if (labels.length > 0 && numeric.length > 0) {
    return { kind: "bar", labelColumn: labels[0]!, valueColumns: numeric };
  }

  // All numeric: two measures against each other is the only honest reading.
  if (numeric.length === 2) {
    return { kind: "scatter", labelColumn: numeric[0]!, valueColumns: [numeric[1]!] };
  }

  return null;
}

/** Why no chart is on offer -- shown so the absence is explained, not silent. */
export function whyNoChart(preview: ResultPreview): string {
  const { columns, rows } = preview;
  if (rows.length === 0) return "no rows came back, so there is nothing to chart";
  if (rows.length < 2) return "a single row is a number, not a chart";
  if (columns.length < 2) return "one column has nothing to plot against";
  if (rows.length > MAX_CATEGORIES) return `${rows.length} rows is more than a chart can show`;
  return "no numeric column to measure";
}
