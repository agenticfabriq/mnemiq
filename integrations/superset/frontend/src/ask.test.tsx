import React from "react";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PanelApi } from "./api";
import { askQuestion } from "./ask";
import { AnswerCard } from "./views/AnswerCard";
import { userTurn, type AnswerPayload, type Turn } from "./shared";

const NOT_A_BUILDER = "Superset refused this user's queries. Building with mnemiq needs Superset's Alpha and sql_lab roles.";

const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;

/** A body that sends `chunks` and then ends, or fails with `cut` after them. */
function body(chunks: string[], cut?: Error): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(bytes.encode(chunk));
      if (cut) controller.error(cut);
      else controller.close();
    },
  });
}

const stream = (chunks: string[], cut?: Error) =>
  new Response(body(chunks, cut), { status: 200, headers: { "content-type": "text/event-stream" } });

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

const ANSWERED = [
  frame({ type: "RUN_STARTED", runId: "r1" }),
  frame({ type: "TEXT_MESSAGE_CONTENT", messageId: "m", delta: "There are 412 open claims." }),
  frame({ type: "CUSTOM", name: "mnemiq.answer", value: answer() }),
  frame({ type: "RUN_FINISHED", runId: "r1" }),
];

/** Asks `question` with `fetchFn`, the turns kept as the panel's store keeps them. */
async function ask(fetchFn: typeof fetch, question = "how many claims are open?", prior: Turn[] = []) {
  let turns = prior;
  await askQuestion(new PanelApi("", async () => "the-csrf-token", fetchFn), question, prior,
    (fn) => { turns = fn(turns); }, new AbortController().signal);
  return turns;
}

describe("asking", () => {
  it("an ask streams through the forwarder and ends complete", async () => {
    const fetchFn = vi.fn<typeof fetch>(async () => stream(ANSWERED));
    const turns = await ask(fetchFn);
    expect(turns.map((t) => [t.role, t.status])).toEqual([["user", "complete"], ["assistant", "complete"]]);
    expect(turns[1].answer?.answer).toBe("There are 412 open claims.");
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("/extensions/agenticfabriq/mnemiq/v1/chat");
    const headers = new Headers(init!.headers);
    expect(headers.get("X-CSRFToken")).toBe("the-csrf-token");
    expect(headers.get("Accept")).toBe("text/event-stream");
    expect(headers.has("Authorization")).toBe(false);
    expect(JSON.parse(String(init!.body))).toEqual({ question: "how many claims are open?", mode: "thinking", history: [] });
  });

  it("a refusal before the stream starts opens its own error turn", async () => {
    const turns = await ask(vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify({ detail: NOT_A_BUILDER }), { status: 403, headers: { "content-type": "application/json" } })));
    expect(turns.map((t) => [t.role, t.status])).toEqual([["user", "complete"], ["assistant", "error"]]);
    expect(turns[1].error).toBe(NOT_A_BUILDER);
  });

  it("a stream cut mid-answer ends as an error, never running", async () => {
    const cut = await ask(vi.fn<typeof fetch>(async () => stream(ANSWERED.slice(0, 2), new TypeError("network error"))));
    expect(cut.map((t) => t.status)).toEqual(["complete", "error"]);
    expect(cut[1].errorCode).toBe("connection-ended");
    // A stream that just stops, with no end frame, ends the same way.
    const stopped = await ask(vi.fn<typeof fetch>(async () => stream(ANSWERED.slice(0, 2))));
    expect(stopped.map((t) => t.status)).toEqual(["complete", "error"]);
    // And a forwarder that answered 504 before anything was streamed.
    const timedOut = await ask(vi.fn<typeof fetch>(async () => new Response(
      JSON.stringify({ detail: "mnemiq took too long to answer" }), { status: 504, headers: { "content-type": "application/json" } })));
    expect(timedOut.map((t) => t.status)).toEqual(["complete", "error"]);
    expect(timedOut[1].error).toBe("mnemiq took too long to answer");
  });

  it("the history an ask sends carries no rows", async () => {
    const prior: Turn[] = [userTurn("how many claims are open?"),
      { id: "a1", role: "assistant", text: "", status: "complete",
        answer: answer({ grant_fingerprint: "fp" } as Partial<AnswerPayload>) }];
    const fetchFn = vi.fn<typeof fetch>(async () => stream(ANSWERED));
    await ask(fetchFn, "and by region?", prior);
    const sent = JSON.parse(String(fetchFn.mock.calls[0][1]!.body));
    expect(sent.history).toEqual([{ question: "how many claims are open?", sql: "SELECT count(*) AS open_claims FROM claim",
      tables_used: ["claim"], columns: ["open_claims"], rows: [], grant_fingerprint: "fp" }]);
  });
});

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

describe("the composer", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("asks from the composer and from a starter, one ask at a time, and shows each answer", async () => {
    const { fireEvent } = await import("@testing-library/react");
    const { Panel } = await import("./Panel");
    const { host } = await import("./test/core");
    const { getSnapshot, resetStore } = await import("./store");
    host.reset();
    resetStore();
    history.replaceState(null, "", "/chart/list/");
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const chats: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      if (String(url).endsWith("/v1/schema")) {
        return new Response(JSON.stringify({ tables: [], starters: ["How many claims are open?"] }),
          { status: 200, headers: { "content-type": "application/json" } });
      }
      chats.push(JSON.parse(String(init!.body)).question);
      if (chats.length === 1) await held;
      return stream(ANSWERED);
    }));
    render(<Panel />);
    fireEvent.click(await screen.findByRole("button", { name: "How many claims are open?" }));
    const box = screen.getByRole("textbox", { name: "Ask mnemiq" });
    fireEvent.change(box, { target: { value: "and by region?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    // The first ask still runs: the second is not sent.
    await new Promise((done) => setTimeout(done, 20));
    expect(chats).toEqual(["How many claims are open?"]);
    release();
    await screen.findByText("There are 412 open claims.");
    fireEvent.change(box, { target: { value: "and by region?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await vi.waitFor(() => expect(getSnapshot().turns.filter((t) => t.role === "assistant" && t.status === "complete")).toHaveLength(2));
    expect(chats).toEqual(["How many claims are open?", "and by region?"]);
  });
});
