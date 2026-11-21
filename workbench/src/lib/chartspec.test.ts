import { describe, expect, it } from "vitest";

import { TABLE_ANSWER } from "../test/fixtures";
import { chartSpecFor, isLabelColumn, isNumeric, isTemporal, whyNoChart } from "./chartspec";
import { compact } from "./chartoption";
import type { Cell, ResultPreview } from "./types";

const preview = (columns: string[], rows: Cell[][]): ResultPreview => ({
  columns,
  rows,
  row_count: rows.length,
  truncated: false,
});

describe("column typing", () => {
  it("counts numerics that arrived as strings", () => {
    expect(isNumeric(["1000.00", "1100.00"])).toBe(true);
    expect(isNumeric([1, 2, 3])).toBe(true);
    expect(isNumeric(["west", "east"])).toBe(false);
  });

  it("does not treat an all-null column as numeric", () => {
    expect(isNumeric([null, null])).toBe(false);
  });

  it("ignores nulls among real numbers", () => {
    expect(isNumeric([1, null, 3])).toBe(true);
  });

  it("recognises dates and timestamps, not bare numbers", () => {
    expect(isTemporal(["2024-04-14", "2024-05-01"])).toBe(true);
    expect(isTemporal(["2024-04-14 10:00:00", "2024-05-01 11:00:00"])).toBe(true);
    expect(isTemporal(["2024", "2025"])).toBe(false);
  });
});

describe("chartSpecFor", () => {
  it("plots a group-by as a bar chart", () => {
    const spec = chartSpecFor(preview(["status", "n"], [["closed", 25263], ["open", 3158]]));
    expect(spec).toEqual({ kind: "bar", labelColumn: "status", valueColumns: ["n"] });
  });

  it("plots a date series as a line", () => {
    const spec = chartSpecFor(
      preview(["month", "total"], [["2024-01", 10], ["2024-02", 20], ["2024-03", 15]]),
    );
    expect(spec).toEqual({ kind: "line", labelColumn: "month", valueColumns: ["total"] });
  });

  it("keeps every measure as its own series", () => {
    const spec = chartSpecFor(
      preview(
        ["region", "claims", "premium"],
        [["west", 1, 2], ["east", 3, 4]],
      ),
    );
    expect(spec?.valueColumns).toEqual(["claims", "premium"]);
  });

  it("plots two bare measures against each other", () => {
    const spec = chartSpecFor(preview(["x", "y"], [[1, 2], [3, 4]]));
    expect(spec).toEqual({ kind: "scatter", labelColumn: "x", valueColumns: ["y"] });
  });

  it("charts the captured multi-row result", () => {
    expect(chartSpecFor(TABLE_ANSWER.preview!)).not.toBeNull();
  });
});

describe("refusing to chart", () => {
  it("declines a single row -- that is a number", () => {
    const p = preview(["n"], [[2]]);
    expect(chartSpecFor(p)).toBeNull();
    expect(whyNoChart(p)).toContain("single row");
  });

  it("declines when nothing is measurable", () => {
    const p = preview(["region", "name"], [["west", "a"], ["east", "b"]]);
    expect(chartSpecFor(p)).toBeNull();
    expect(whyNoChart(p)).toContain("numeric");
  });

  it("declines a result with too many categories to read", () => {
    const rows: Cell[][] = Array.from({ length: 100 }, (_, i) => [`c${i}`, i]);
    const p = preview(["policy", "amount"], rows);
    expect(chartSpecFor(p)).toBeNull();
    expect(whyNoChart(p)).toContain("100 rows");
  });

  it("declines a single column", () => {
    expect(chartSpecFor(preview(["n"], [[1], [2]]))).toBeNull();
  });
});

describe("compact axis figures", () => {
  it("abbreviates the magnitudes a claims total actually reaches", () => {
    expect(compact(4149117420)).toBe("4.1B");
    expect(compact(1917098896)).toBe("1.9B");
    expect(compact(2500000)).toBe("2.5M");
    expect(compact(16667)).toBe("16.7k");
  });

  it("leaves readable numbers alone", () => {
    expect(compact(0)).toBe("0");
    expect(compact(3158)).toBe("3,158");
  });

  it("keeps the sign", () => {
    expect(compact(-2500000)).toBe("-2.5M");
  });
});

describe("years and identifiers are labels", () => {
  it("charts a year against a total as a bar labelled by the year", () => {
    const spec = chartSpecFor(preview(["year", "total"], [[2023, 10], [2024, 20], [2025, 15]]));
    expect(spec).toEqual({ kind: "bar", labelColumn: "year", valueColumns: ["total"] });
  });

  it("keeps the region the label and the year out of the values", () => {
    const spec = chartSpecFor(
      preview(["region", "year", "total"], [["west", 2024, 10], ["east", 2024, 20], ["west", 2025, 5]]),
    );
    expect(spec).toEqual({ kind: "bar", labelColumn: "region", valueColumns: ["total"] });
  });

  it("labels the bars by an identifier column", () => {
    const spec = chartSpecFor(preview(["customer_id", "total"], [[101, 10], [102, 20], [103, 5]]));
    expect(spec).toEqual({ kind: "bar", labelColumn: "customer_id", valueColumns: ["total"] });
  });

  it("reads a fiscal year column of four-digit years as a label", () => {
    const spec = chartSpecFor(preview(["fiscal_yr", "revenue"], [[2023, 1.5], [2024, 2.5]]));
    expect(spec).toEqual({ kind: "bar", labelColumn: "fiscal_yr", valueColumns: ["revenue"] });
  });

  it("does not take whole numbers in a year's range for years without a year in the name", () => {
    const spec = chartSpecFor(preview(["score", "total"], [[1950, 2001], [1984, 2010], [2003, 1999]]));
    expect(spec?.kind).toBe("scatter");
  });

  it("names a label column by its identifier word or its year", () => {
    expect(isLabelColumn("id", [1, 2])).toBe(true);
    expect(isLabelColumn("order_no", [1, 2])).toBe(true);
    expect(isLabelColumn("zip_code", [94301])).toBe(true);
    expect(isLabelColumn("customerId", [1])).toBe(true);
    expect(isLabelColumn("phone_number", [5551234])).toBe(true);
    expect(isLabelColumn("year", [2024, null])).toBe(true);
    expect(isLabelColumn("year", [2024.5])).toBe(false);
    expect(isLabelColumn("paid", [10])).toBe(false);
    expect(isLabelColumn("video", [1])).toBe(false);
  });
});

describe("an empty result says it is empty", () => {
  it("says no rows came back when there are none", () => {
    expect(whyNoChart(preview(["region", "total"], []))).toBe("no rows came back, so there is nothing to chart");
  });

  it("keeps a single row a number", () => {
    expect(whyNoChart(preview(["total"], [[3]]))).toBe("a single row is a number, not a chart");
  });
});
