import { render, screen } from "@testing-library/react";
import type { DataMessagePartProps } from "@assistant-ui/react";
import { describe, expect, it } from "vitest";

import { SCALAR_ANSWER } from "../test/fixtures";
import type { Turn } from "../lib/store";
import { SqlBlock } from "./SqlBlock";
import { TurnArtifacts } from "./TurnArtifacts";

/**
 * M51 made the payload's `sql` the statement as asked: under a row or column policy the executed
 * one names the policy's own tables and literals, and stays in the audit trace. The header still
 * said "Executed SQL", which was false exactly when a policy narrowed the answer -- and `narrowed`,
 * which says so, reached the browser and nothing read it.
 *
 * `columns` is "the query referenced a column name this table masks" (and may name a table whose
 * column the caller never read, where ownership does not resolve), so the note says "masked columns
 * referenced on", never "columns masked on".
 */
const SQL = "SELECT COUNT(*) FROM claim JOIN person ON person.id = claim.person_id";

describe("SqlBlock", () => {
  it("names what a policy narrowed, before the statement is opened", () => {
    render(
      <SqlBlock
        sql={SQL}
        narrowed={[
          { object: "claim", rows: true, columns: false },
          { object: "person", rows: false, columns: true },
        ]}
      />,
    );
    expect(screen.getByRole("button", { name: /SQL as asked/ })).toBeInTheDocument();
    expect(screen.queryByText(/Executed SQL/)).not.toBeInTheDocument();
    expect(screen.getByText("Rows withheld from: claim.")).toBeInTheDocument();
    expect(screen.getByText("Masked columns referenced on: person.")).toBeInTheDocument();
    expect(screen.getByText(/The statement that ran also applied the policy/)).toBeInTheDocument();
    // Collapsed: the policy is stated without opening the statement.
    expect(screen.queryByText(SQL)).not.toBeInTheDocument();
  });

  it("lists an object under each flag it carries, in the engine's order", () => {
    render(
      <SqlBlock
        sql={SQL}
        narrowed={[
          { object: "claim", rows: true, columns: true },
          { object: "policy", rows: true, columns: false },
        ]}
      />,
    );
    expect(screen.getByText("Rows withheld from: claim, policy.")).toBeInTheDocument();
    expect(screen.getByText("Masked columns referenced on: claim.")).toBeInTheDocument();
  });

  it("calls the statement executed when the policy narrowed nothing", () => {
    // Every rewrite records a narrowing, so with none the statement as asked IS the one that ran.
    render(<SqlBlock sql={SQL} narrowed={[]} />);
    expect(screen.getByRole("button", { name: /Executed SQL/ })).toBeInTheDocument();
    expect(screen.queryByText(/withheld|masked|policy/i)).not.toBeInTheDocument();
  });

  it("claims no narrowing when the engine did not evaluate one or did not report it", () => {
    // `null` is "not evaluated"; absent is an engine from before the field. Neither is a claim.
    for (const narrowed of [null, undefined]) {
      const { unmount } = render(<SqlBlock sql={SQL} narrowed={narrowed} />);
      expect(screen.getByRole("button", { name: /Executed SQL/ })).toBeInTheDocument();
      expect(screen.queryByText(/withheld|masked|policy/i)).not.toBeInTheDocument();
      unmount();
    }
  });
});

describe("TurnArtifacts", () => {
  it("hands the answer's narrowing to the SQL block", () => {
    const turn: Turn = {
      id: "t1",
      role: "assistant",
      text: SCALAR_ANSWER.answer,
      status: "complete",
      answer: { ...SCALAR_ANSWER, narrowed: [{ object: "claim", rows: true, columns: false }] },
    };
    // Only `data` is read; the rest of the part props belong to assistant-ui's runtime.
    render(<TurnArtifacts {...({ data: turn } as unknown as DataMessagePartProps<Turn>)} />);
    expect(screen.getByText("Rows withheld from: claim.")).toBeInTheDocument();
  });
});
