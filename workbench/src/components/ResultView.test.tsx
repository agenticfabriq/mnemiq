import { render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it } from "vitest";

import type { ResultPreview } from "../lib/types";
import { ResultView } from "./ResultView";

// A label column and a number: chartable, so the chart and its caption are the default view.
const chartable = (extra: Partial<ResultPreview>): ResultPreview => ({
  columns: ["status", "n"],
  rows: [["closed", 25263], ["open", 3158]],
  row_count: 2,
  truncated: false,
  ...extra,
});

describe("ResultView", () => {
  // The chart sizes itself with ResizeObserver, which jsdom does not provide; a no-op is enough to
  // render the caption these tests read.
  beforeAll(() => {
    globalThis.ResizeObserver ??= class {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver;
  });

  it("says on the chart that the query stopped at the row limit (M118)", () => {
    render(<ResultView preview={chartable({ row_count: 1000, truncated: true, capped: true })} />);
    expect(screen.getByText(/2 of 1,000\+ rows/)).toBeInTheDocument();
    expect(screen.getByText(/the query stopped at the 1,000-row limit/)).toBeInTheDocument();
  });

  it("tells the CSV download there may be more rows, not that there are", () => {
    render(<ResultView preview={chartable({ row_count: 1000, truncated: true, capped: true })} />);
    const csv = screen.getByRole("button", { name: /CSV/ });
    expect(csv.getAttribute("title")).toMatch(/1,000-row limit, so there may be more/);
  });

  it("says it for a capped chart even when every row fits on screen", () => {
    render(<ResultView preview={chartable({ row_count: 1000, truncated: false, capped: true })} />);
    expect(screen.getByText(/1,000\+ rows/)).toBeInTheDocument();
    expect(screen.queryByText(/ of 1,000/)).not.toBeInTheDocument();  // the uncut-preview caption ran
    expect(screen.getByText(/row limit/)).toBeInTheDocument();
  });

  it("does not claim a row limit for a long preview the engine did not cut", () => {
    render(<ResultView preview={chartable({ row_count: 999, truncated: true })} />);
    expect(screen.getByText(/2 of 999 rows/)).toBeInTheDocument();
    expect(screen.queryByText(/row limit/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /CSV/ }).getAttribute("title")).toMatch(/not all 999/);
  });

  it("leaves an uncut chart's caption as it was", () => {
    render(<ResultView preview={chartable({})} />);
    expect(screen.getByText(/2 rows/)).toBeInTheDocument();
    expect(screen.queryByText(/row limit/)).not.toBeInTheDocument();
  });
});
