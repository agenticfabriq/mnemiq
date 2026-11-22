import React from "react";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AnswerCard } from "./views/AnswerCard";
import { type AnswerPayload, type Turn } from "./shared";


function answer(fields: Partial<AnswerPayload> = {}): AnswerPayload {
  return {
    answer: "There are 412 open claims.", deferred: false, failed: false, reason_code: null, mode: "thinking",
    cached: false, agreement: null, judge_engaged: null, judge_override: null, judge_fell_back: null,
    candidates_executed: null, attempts: null, corrected: null, sql: "SELECT count(*) AS open_claims FROM claim",
    tables_used: ["claim"], enrichment_version: null, timing: null,
    preview: { columns: ["open_claims"], rows: [[412]], row_count: 1, truncated: false },
    ...fields,
  } as AnswerPayload;
}


describe("the answer card", () => {
  afterEach(cleanup);

  it("a blank cell is never shown as 0", () => {
    const turn: Turn = { id: "a", role: "assistant", text: "", status: "complete",
      answer: answer({ preview: { columns: ["region", "claims"], rows: [["North", 0], [null, null]], row_count: 2, truncated: false } }) };
    render(<AnswerCard turn={turn} />);
    const cells = within(screen.getByRole("table")).getAllByRole("cell").map((c) => c.textContent);
    expect(cells).toEqual(["North", "0", "—", "—"]);
  });

  it("shows the sentence, and the SQL folded", () => {
    render(<AnswerCard turn={{ id: "a", role: "assistant", text: "There are 412 open claims.", status: "complete", answer: answer() }} />);
    expect(screen.getByText("There are 412 open claims.")).toBeTruthy();
    const folded = screen.getByText("SQL").closest("details")!;
    expect(folded.open).toBe(false);
    expect(folded.textContent).toContain("SELECT count(*) AS open_claims FROM claim");
  });

  it("a declined answer says why and what to do next", () => {
    render(<AnswerCard turn={{ id: "a", role: "assistant", text: "", status: "complete",
      answer: answer({ deferred: true, reason_code: "authorization", answer: "I can't answer that.", preview: null, sql: null }) }} />);
    expect(screen.getByText("Out of scope for this identity")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("a cut connection says so", () => {
    render(<AnswerCard turn={{ id: "a", role: "assistant", text: "", status: "error",
      error: "The connection ended before the answer was complete; ask again.", errorCode: "connection-ended" }} />);
    expect(screen.getByRole("alert").textContent).toContain("The connection ended");
  });
});

describe("the composer's keys", () => {
  afterEach(cleanup);

  it("Enter while composing with an input method does not send", async () => {
    const { Composer } = await import("./views/Composer");
    const asked: string[] = [];
    render(<Composer busy={false} placeholder="Ask" onSend={(text) => { asked.push(text); return true; }} />);
    const box = screen.getByRole("textbox", { name: "Message mnemiq" });
    const { fireEvent } = await import("@testing-library/react");
    fireEvent.change(box, { target: { value: "請問有多少" } });
    fireEvent.keyDown(box, { key: "Enter", code: "Enter", keyCode: 13, isComposing: true });
    expect(asked).toEqual([]);
    fireEvent.keyDown(box, { key: "Enter", code: "Enter", keyCode: 13 });
    expect(asked).toEqual(["請問有多少"]);
  });
});
