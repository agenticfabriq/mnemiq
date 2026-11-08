import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Turn } from "../lib/store";
import { TurnArtifacts } from "./TurnArtifacts";

const rows = {
  sql: "SELECT name FROM claim",
  tables_used: ["claim"],
  narrowed: [],
  preview: { columns: ["name"], rows: [["only-row-value"]], row_count: 1, truncated: false },
  notes: ["These results stop at the 1,000-row limit, so there may be more."],
  rows_ms: 9000,
};

const show = (turn: Partial<Turn>) => {
  const data: Turn = { id: "a", role: "assistant", text: "", status: "running", ...turn };
  // Only `data` is read; the rest of assistant-ui's part props do not matter here.
  const props = { data } as unknown as Parameters<typeof TurnArtifacts>[0];
  return render(<TurnArtifacts {...props} />);
};

describe("TurnArtifacts", () => {
  it("shows the verified rows and their notes while the sentence is written", () => {
    show({ result: rows });
    expect(screen.getByText("only-row-value")).toBeInTheDocument();
    expect(screen.getByText(rows.notes[0]!)).toBeInTheDocument();
    expect(screen.getByText(/Writing the answer/)).toBeInTheDocument();
  });

  it("says the engine is still serving when a run fails on the engine", () => {
    show({ status: "error", error: "the run failed" });
    expect(screen.getByText(/The engine is still serving/)).toBeInTheDocument();
  });

  it("says a service problem is one, rather than that the engine is still serving", () => {
    // A 503 or a request that never got an answer: the engine is exactly what is not serving.
    show({ status: "error", error: "The service is unavailable.", errorCode: "unavailable" });
    expect(screen.getByText(/service problem/)).toBeInTheDocument();
    expect(screen.queryByText(/still serving/)).not.toBeInTheDocument();
  });

  it("shows only progress before the rows arrive", () => {
    show({});
    expect(screen.queryByText("only-row-value")).not.toBeInTheDocument();
    expect(screen.getByText(/Working/)).toBeInTheDocument();
  });
});
