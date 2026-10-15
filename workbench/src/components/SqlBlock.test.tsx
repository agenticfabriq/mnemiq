import { render, screen } from "@testing-library/react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { SCALAR_ANSWER } from "../test/fixtures";
import { useWorkbench } from "../lib/runtime";
import { save } from "../lib/threads";
import { SqlBlock } from "./SqlBlock";
import { Thread } from "./Thread";

/**
 * M51 made the payload's `sql` the statement as asked: under a row or column policy the executed
 * one names the policy's own tables and literals, and stays in the audit trace. The header still
 * said "Executed SQL", which was false exactly when a policy narrowed the answer -- and `narrowed`,
 * which says so, reached the browser and nothing read it.
 *
 * The note claims only what the flags record (`Narrowing` in src/mnemiq/sql/rls.py). `rows` is "a
 * row filter was applied to it": the engine never measures whether the filter removed anything, so
 * the note says a filter was applied, never that rows were withheld. `columns` is "the query
 * referenced a column name this table masks", and where ownership does not resolve it can name a
 * table whose column the caller never read, so the note names column NAMES, not masked columns
 * (Codex's review of #145).
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
    expect(screen.getByText("Row filter applied to: claim.")).toBeInTheDocument();
    expect(screen.getByText("Masked column names referenced, on: person.")).toBeInTheDocument();
    expect(screen.getByText(/The statement that ran also applied the policy/)).toBeInTheDocument();
    // Collapsed: the policy is stated without opening the statement.
    expect(screen.queryByText(SQL)).not.toBeInTheDocument();
  });

  it("claims a filter ran, never that it removed rows, nor that a named table's column was read", () => {
    // A filter every row passes sets `rows` exactly as one that removes most of them does, and an
    // unresolved column name sets `columns` on every table that masks that name.
    render(
      <SqlBlock
        sql={SQL}
        narrowed={[
          { object: "claim", rows: true, columns: true },
          { object: "person", rows: false, columns: true },
        ]}
      />,
    );
    expect(screen.queryByText(/withheld|removed|excluded/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/masked columns|columns (were )?masked/i)).not.toBeInTheDocument();
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
    expect(screen.getByText("Row filter applied to: claim, policy.")).toBeInTheDocument();
    expect(screen.getByText("Masked column names referenced, on: claim.")).toBeInTheDocument();
  });

  it("calls the statement executed when the policy narrowed nothing", () => {
    // Every rewrite records a narrowing, so with none the statement as asked IS the one that ran.
    render(<SqlBlock sql={SQL} narrowed={[]} />);
    expect(screen.getByRole("button", { name: /Executed SQL/ })).toBeInTheDocument();
    expect(screen.queryByText(/filter|masked|policy/i)).not.toBeInTheDocument();
  });

  it("claims no narrowing when the engine did not evaluate one or did not report it", () => {
    // `null` is "not evaluated"; absent is an engine from before the field. Neither is a claim.
    for (const narrowed of [null, undefined]) {
      const { unmount } = render(<SqlBlock sql={SQL} narrowed={narrowed} />);
      expect(screen.getByRole("button", { name: /Executed SQL/ })).toBeInTheDocument();
      expect(screen.queryByText(/filter|masked|policy/i)).not.toBeInTheDocument();
      unmount();
    }
  });
});

function Workbench() {
  const { runtime } = useWorkbench();
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread starters={[]} onPick={() => {}} />
    </AssistantRuntimeProvider>
  );
}

describe("the transcript", () => {
  // assistant-ui's viewport observes its own size; jsdom has no ResizeObserver.
  beforeAll(() => {
    globalThis.ResizeObserver ??= class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });
  afterEach(() => localStorage.clear());

  it("carries an answer's narrowing from the stored turn to the SQL block", () => {
    // The path a real answer takes: the stored turn, the runtime's message conversion, the
    // `mnemiq.turn` data part dispatched by name, TurnArtifacts, SqlBlock -- no part props faked.
    save([
      {
        id: "thread-1",
        title: "policies",
        createdAt: 0,
        turns: [
          { id: "q1", role: "user", text: "How many policies?", status: "complete" },
          {
            id: "a1",
            role: "assistant",
            text: SCALAR_ANSWER.answer,
            status: "complete",
            answer: { ...SCALAR_ANSWER, narrowed: [{ object: "policy", rows: true, columns: false }] },
          },
        ],
      },
    ]);
    render(<Workbench />);
    expect(screen.getByRole("button", { name: /SQL as asked/ })).toBeInTheDocument();
    expect(screen.getByText("Row filter applied to: policy.")).toBeInTheDocument();
  });
});
