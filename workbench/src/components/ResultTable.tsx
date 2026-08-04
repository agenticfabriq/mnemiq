/**
 * The executed result, from the preview the engine already carried back. The
 * workbench never re-runs SQL to draw a table -- that would dodge the governed read
 * path and could disagree with the answer above it.
 */

import type { Cell, ResultPreview } from "../lib/types";

const isNumeric = (value: Cell) =>
  typeof value === "number" ||
  (typeof value === "string" && value !== "" && !Number.isNaN(Number(value)));

function CellView({ value }: { value: Cell }) {
  if (value === null || value === undefined) {
    return <span className="text-graphite/60 italic">null</span>;
  }
  if (typeof value === "boolean") return <>{String(value)}</>;
  return <>{String(value)}</>;
}

export function ResultTable({ preview }: { preview: ResultPreview }) {
  const { columns, rows, row_count, truncated } = preview;
  if (columns.length === 0) return null;

  // Right-align a column only when every cell in it reads as a figure.
  const numeric = columns.map((_, i) =>
    rows.length > 0 && rows.every((row) => row[i] === null || isNumeric(row[i] as Cell)),
  );

  return (
    // Hug the content when the result is narrow; scroll inside the border when it is
    // wider than the column, so the page itself never scrolls sideways.
    <figure className="w-fit max-w-full border border-rule">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[12px]">
          <thead>
            <tr className="border-b border-rule bg-sunken">
              {columns.map((column, c) => (
                <th
                  key={column}
                  scope="col"
                  className={`meta px-2.5 py-1.5 whitespace-nowrap ${
                    numeric[c] ? "text-right" : "text-left"
                  }`}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, r) => (
              <tr key={r} className="border-b border-rule/60 last:border-b-0">
                {columns.map((column, c) => (
                  <td
                    key={column}
                    className={`px-2.5 py-1 whitespace-nowrap ${
                      numeric[c] ? "text-right" : ""
                    }`}
                  >
                    <CellView value={(row[c] ?? null) as Cell} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption className="label border-t border-rule px-2.5 py-1.5">
        {truncated
          ? `showing ${rows.length} of ${row_count} rows`
          : `${row_count} ${row_count === 1 ? "row" : "rows"}`}
      </figcaption>
    </figure>
  );
}
