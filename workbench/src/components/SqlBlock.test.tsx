import { render, screen } from "@testing-library/react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { SCALAR_ANSWER } from "../test/fixtures";
import type { AnswerPayload } from "../lib/types";
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
 * What the policy narrowed is the ENGINE's to state: since #150 (M155) it ends every answered
 * result's text with "Row filter applied to: ..." / "Masked column names referenced, on: ...". So
 * the block says only what the answer does not -- that the statement shown is not the one that ran
 * -- and never repeats the list. A refusal carries no trace, hence no `sql` and no `narrowed`, and
 * shows no SQL block at all.
 */
const SQL = "SELECT COUNT(*) FROM claim JOIN person ON person.id = claim.person_id";

describe("SqlBlock", () => {
  it("says the statement shown did not run, before it is opened, when a policy narrowed the answer", () => {
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
    expect(screen.getByText(/The statement that ran also applied the policy/)).toBeInTheDocument();
    // Collapsed: the fact is stated without opening the statement.
    expect(screen.queryByText(SQL)).not.toBeInTheDocument();
  });

  it("leaves naming what was narrowed to the answer, and never claims rows were removed", () => {
    render(<SqlBlock sql={SQL} narrowed={[{ object: "claim", rows: true, columns: true }]} />);
    expect(screen.queryByText(/claim/)).not.toBeInTheDocument();
    expect(screen.queryByText(/withheld|removed|excluded|masked columns/i)).not.toBeInTheDocument();
  });

  it("calls the statement executed when the policy narrowed nothing", () => {
    // Every rewrite records a narrowing, so with none the statement as asked IS the one that ran.
    render(<SqlBlock sql={SQL} narrowed={[]} />);
    expect(screen.getByRole("button", { name: /Executed SQL/ })).toBeInTheDocument();
    expect(screen.queryByText(/policy/i)).not.toBeInTheDocument();
  });

  it("claims no narrowing when the engine did not evaluate one or did not report it", () => {
    // `null` is "not evaluated"; absent is an engine from before the field. Neither is a claim.
    for (const narrowed of [null, undefined]) {
      const { unmount } = render(<SqlBlock sql={SQL} narrowed={narrowed} />);
      expect(screen.getByRole("button", { name: /Executed SQL/ })).toBeInTheDocument();
      expect(screen.queryByText(/policy/i)).not.toBeInTheDocument();
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

  function store(answer: AnswerPayload) {
    save([
      {
        id: "thread-1",
        title: "policies",
        createdAt: 0,
        turns: [
          { id: "q1", role: "user", text: "How many policies?", status: "complete" },
          { id: "a1", role: "assistant", text: answer.answer, status: "complete", answer },
        ],
      },
    ]);
  }
  const NARROWED = [{ object: "policy", rows: true, columns: false }];

  it("states an answered result's narrowing once: in the answer, not again under the SQL", () => {
    // The path a real answer takes: the stored turn, the runtime's message conversion, the
    // `mnemiq.turn` data part dispatched by name, TurnArtifacts, SqlBlock -- no part props faked.
    // The text ends with the engine's own sentence, as an answered result's does since #150.
    store({ ...SCALAR_ANSWER, answer: "west: 16,666 policies.\n\nRow filter applied to: policy.", narrowed: NARROWED });
    render(<Workbench />);
    expect(screen.getByRole("button", { name: /SQL as asked/ })).toBeInTheDocument();
    expect(screen.getByText(/The statement that ran also applied the policy/)).toBeInTheDocument();
    expect(screen.getAllByText(/Row filter applied to: policy\./)).toHaveLength(1);
  });
});
