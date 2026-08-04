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
