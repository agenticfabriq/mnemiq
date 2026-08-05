/**
 * The result, as a table or as a chart.
 *
 * The table is the ground truth and stays one click away always -- it is also the
 * required relief for the chart palette's lighter slots, which fall under 3:1
 * against the light surface. When the shape cannot be charted honestly the toggle
 * says why rather than disappearing.
 */

import { useState } from "react";

import { chartSpecFor, whyNoChart } from "../lib/chartspec";
import { downloadCsv } from "../lib/csv";
import type { ResultPreview } from "../lib/types";
import { ResultChart } from "./ResultChart";
import { ResultTable } from "./ResultTable";

export function ResultView({ preview }: { preview: ResultPreview }) {
  const spec = chartSpecFor(preview);
  const [view, setView] = useState<"table" | "chart">("table");
  const [measure, setMeasure] = useState<string | null>(null);
  const showing = spec ? view : "table";
  const shown = measure && spec?.valueColumns.includes(measure) ? measure : spec?.valueColumns[0];

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <div role="radiogroup" aria-label="Result view" className="flex border border-rule">
          {(["table", "chart"] as const).map((option) => (
            <button
              key={option}
              type="button"
              role="radio"
              aria-checked={showing === option}
              disabled={option === "chart" && !spec}
              title={option === "chart" && !spec ? whyNoChart(preview) : undefined}
              onClick={() => setView(option)}
              className={`label border-r border-rule px-2 py-1 last:border-r-0 disabled:opacity-40 ${
                showing === option ? "bg-ink text-paper" : "hover:text-ink"
              }`}
            >
              {option}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => downloadCsv(preview)}
          title={
            preview.truncated
              ? `Downloads the ${preview.rows.length} rows shown, not all ${preview.row_count}`
              : `Downloads all ${preview.row_count} rows`
          }
          className="label border border-rule px-2 py-1 hover:border-brass hover:text-brass"
        >
          {preview.truncated ? `CSV (${preview.rows.length} shown)` : "CSV"}
        </button>

        {!spec && <span className="meta">{whyNoChart(preview)}</span>}

        {/* One measure at a time -- see chartoption.ts for why there is no second axis. */}
        {showing === "chart" && spec && spec.valueColumns.length > 1 && (
          <label className="meta flex items-center gap-1.5">
            <span className="label">Measure</span>
            <select
              value={shown}
              onChange={(event) => setMeasure(event.target.value)}
              className="border border-rule bg-transparent px-1 py-0.5 text-[11px]"
            >
              {spec.valueColumns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {showing === "chart" && spec && shown ? (
        <figure className="border border-rule px-2 py-2">
          <ResultChart spec={spec} preview={preview} measure={shown} />
          <figcaption className="meta px-0.5 pt-1">
            {shown} by {spec.labelColumn}
            {" · "}
            {preview.truncated
              ? `${preview.rows.length} of ${preview.row_count} rows`
              : `${preview.row_count} ${preview.row_count === 1 ? "row" : "rows"}`}
          </figcaption>
        </figure>
      ) : (
        <ResultTable preview={preview} />
      )}
    </div>
  );
}
