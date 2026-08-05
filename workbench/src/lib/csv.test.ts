import { describe, expect, it } from "vitest";

import { TABLE_ANSWER } from "../test/fixtures";
import { toCsv } from "./csv";
import type { Cell, ResultPreview } from "./types";

const preview = (columns: string[], rows: Cell[][]): ResultPreview => ({
  columns,
  rows,
  row_count: rows.length,
  truncated: false,
});

describe("toCsv", () => {
  it("writes a header and one line per row", () => {
    expect(toCsv(preview(["a", "b"], [[1, "x"], [2, "y"]]))).toBe("a,b\r\n1,x\r\n2,y");
  });

  it("quotes separators, quotes and newlines", () => {
    const csv = toCsv(preview(["note"], [['he said "hi", loudly']]));
    expect(csv).toBe('note\r\n"he said ""hi"", loudly"');
  });

  it("keeps an embedded newline inside one quoted field", () => {
    expect(toCsv(preview(["note"], [["line1\nline2"]]))).toBe('note\r\n"line1\nline2"');
  });

  it("writes a null as an empty field, not the word null", () => {
    expect(toCsv(preview(["a", "b"], [[null, 1]]))).toBe("a,b\r\n,1");
  });

  it("exports the captured result verbatim", () => {
    const csv = toCsv(TABLE_ANSWER.preview!);
    const lines = csv.split("\r\n");
    expect(lines[0]).toBe(TABLE_ANSWER.preview!.columns.join(","));
    expect(lines).toHaveLength(TABLE_ANSWER.preview!.rows.length + 1);
  });
});
