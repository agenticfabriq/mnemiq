import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SCALAR_ANSWER, TABLE_ANSWER } from "../test/fixtures";
import type { ResultPreview } from "../lib/types";
import { ResultTable } from "./ResultTable";

describe("ResultTable", () => {
  it("renders the captured result with its columns and rows", () => {
    render(<ResultTable preview={TABLE_ANSWER.preview!} />);
    expect(screen.getByRole("columnheader", { name: "claim_identifier" })).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(9); // header + 8
    expect(screen.getByText("1000.00")).toBeInTheDocument();
  });

  it("states the row count when nothing was truncated", () => {
    render(<ResultTable preview={TABLE_ANSWER.preview!} />);
    expect(screen.getByText("8 rows")).toBeInTheDocument();
  });

  it("says how much of the result is on screen when truncated", () => {
    const preview: ResultPreview = {
      columns: ["n"],
      rows: Array.from({ length: 100 }, (_, i) => [i]),
      row_count: 820,
      truncated: true,
    };
    render(<ResultTable preview={preview} />);
    expect(screen.getByText("showing 100 of 820 rows")).toBeInTheDocument();
  });

  it("says the query stopped at the row limit when the engine cut it off", () => {
    render(
      <ResultTable
        preview={{ columns: ["n"], rows: [[1], [2]], row_count: 1000, truncated: true, capped: true }}
      />,
    );
    expect(screen.getByText(/showing 2 of 1,000\+ rows/)).toBeInTheDocument();
    expect(screen.getByText(/the query stopped at the 1,000-row limit/)).toBeInTheDocument();
  });

  it("says it for a capped result even when every row fits on screen", () => {
    // capped and truncated are different facts: the engine's limit against the preview's.
    render(<ResultTable preview={{ columns: ["n"], rows: [[1]], row_count: 1000, truncated: false, capped: true }} />);
    expect(screen.getByText(/1,000\+ rows/)).toBeInTheDocument();
    expect(screen.queryByText(/showing/)).not.toBeInTheDocument();  // the uncut-preview caption ran
    expect(screen.getByText(/row limit/)).toBeInTheDocument();
  });

  it("does not call a long preview capped when the engine did not cut it", () => {
    render(<ResultTable preview={{ columns: ["n"], rows: [[1]], row_count: 999, truncated: true }} />);
    expect(screen.getByText(/showing 1 of 999 rows/)).toBeInTheDocument();
    expect(screen.queryByText(/row limit/)).not.toBeInTheDocument();
  });

  it("singularises a one-row result", () => {
    render(<ResultTable preview={SCALAR_ANSWER.preview!} />);
    expect(screen.getByText("1 row")).toBeInTheDocument();
  });

  it("shows a null as null rather than as blank", () => {
    const preview: ResultPreview = {
      columns: ["loss_date"],
      rows: [[null]],
      row_count: 1,
      truncated: false,
    };
    render(<ResultTable preview={preview} />);
    expect(screen.getByText("null")).toBeInTheDocument();
  });

  it("renders nothing when the result has no columns", () => {
    const { container } = render(
      <ResultTable preview={{ columns: [], rows: [], row_count: 0, truncated: false }} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
