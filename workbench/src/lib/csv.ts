/**
 * The result rows, as CSV.
 *
 * Exports exactly what the engine returned -- the bounded preview -- and the caller
 * is told so. Silently handing someone a 100-row file when their query matched
 * 30,000 would be the same class of lie as a fabricated chart.
 */

import type { Cell, ResultPreview } from "./types";

const quote = (value: Cell): string => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

export function toCsv(preview: ResultPreview): string {
  const rows = preview.rows.map((row) =>
    preview.columns.map((_, i) => quote((row[i] ?? null) as Cell)).join(","),
  );
  return [preview.columns.map((c) => quote(c)).join(","), ...rows].join("\r\n");
}

/** Excel reads a UTF-8 file as latin-1 without a byte-order mark and mangles it. */
export const csvBlob = (preview: ResultPreview): Blob =>
  new Blob([`﻿${toCsv(preview)}`], { type: "text/csv;charset=utf-8;" });

export function downloadCsv(preview: ResultPreview, filename = "result.csv"): void {
  const url = URL.createObjectURL(csvBlob(preview));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
