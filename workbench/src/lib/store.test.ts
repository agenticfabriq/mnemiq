import { describe, expect, it } from "vitest";

import { DEFERRAL, SCALAR_ANSWER } from "../test/fixtures";
import {
  ANSWER_EVENT,
  CONNECTION_ENDED,
  CONNECTION_ENDED_CODE,
  RESULT_EVENT,
  closeRun,
  isRunning,
  reduce,
  settle,
  userTurn,
  type Turn,
} from "./store";
import { streamChat } from "./transport";
import type { AguiEvent } from "./types";

const run = (events: AguiEvent[], from: Turn[] = []) => events.reduce(reduce, from);

const answered = (payload = SCALAR_ANSWER): AguiEvent[] => [
  { type: "RUN_STARTED", runId: "r1" },
  { type: "TEXT_MESSAGE_START", messageId: "m1", role: "assistant" },
  { type: "TEXT_MESSAGE_CONTENT", messageId: "m1", delta: payload.answer },
  { type: "TEXT_MESSAGE_END", messageId: "m1" },
  { type: "CUSTOM", name: ANSWER_EVENT, value: payload },
  { type: "RUN_FINISHED", runId: "r1" },
];

describe("reduce", () => {
  it("builds one complete assistant turn from the happy sequence", () => {
    const turns = run(answered());
    expect(turns).toHaveLength(1);
    expect(turns[0]!.role).toBe("assistant");
    expect(turns[0]!.text).toBe("There are 2 claims.");
    expect(turns[0]!.status).toBe("complete");
    expect(turns[0]!.answer?.sql).toContain("SELECT COUNT");
  });

  it("keeps the deferral reason on the payload rather than inventing a result", () => {
    const turns = run(answered(DEFERRAL));
    expect(turns[0]!.answer?.deferred).toBe(true);
    expect(turns[0]!.answer?.reason_code).toBe("unanswerable");
    expect(turns[0]!.answer?.preview).toBeNull();
    expect(turns[0]!.status).toBe("complete");
  });

  it("accumulates several content deltas in order", () => {
    const turns = run([
      { type: "RUN_STARTED", runId: "r1" },
      { type: "TEXT_MESSAGE_CONTENT", messageId: "m1", delta: "There are " },
      { type: "TEXT_MESSAGE_CONTENT", messageId: "m1", delta: "2 claims." },
    ]);
    expect(turns[0]!.text).toBe("There are 2 claims.");
    expect(turns[0]!.status).toBe("running");
  });

  it("marks the turn errored and carries the message", () => {
    const turns = run([
      { type: "RUN_STARTED", runId: "r1" },
      { type: "RUN_ERROR", message: "boom", runId: "r1" },
    ]);
    expect(turns[0]!.status).toBe("error");
    expect(turns[0]!.error).toBe("boom");
    expect(turns[0]!.answer).toBeUndefined();
    expect(turns[0]!.errorCode).toBeUndefined();
  });

  it("keeps the error's code, which says what kind of failure it was", () => {
    const turns = run([
      { type: "RUN_STARTED", runId: "" },
      { type: "RUN_ERROR", message: "The service is unavailable.", code: "unavailable" },
    ]);
    expect(turns[0]!.errorCode).toBe("unavailable");
  });

  it("appends after a user turn without disturbing it", () => {
    const turns = run(answered(), [userTurn("how many claims are there?")]);
    expect(turns).toHaveLength(2);
    expect(turns[0]!.role).toBe("user");
    expect(turns[0]!.text).toBe("how many claims are there?");
  });

  it("ignores an event type it does not know", () => {
    const before = run(answered());
    const after = reduce(before, { type: "STEP_STARTED" } as AguiEvent);
    expect(after).toBe(before);
  });

  it("keeps the verified rows that arrive before the answer", () => {
    const rows = { sql: "SELECT n FROM claim", tables_used: ["claim"], narrowed: [], preview: null,
      notes: ["a note"], rows_ms: 10 };
    const turns = [{ type: "RUN_STARTED", runId: "r1" }, { type: "CUSTOM", name: RESULT_EVENT, value: rows }]
      .reduce(reduce, [] as Turn[]);
    expect(turns[0]!.result).toEqual(rows);
    expect(turns[0]!.answer).toBeUndefined();
    expect(turns[0]!.status).toBe("running");
  });

  it("lands the answer after the rows and keeps both", () => {
    const rows = { sql: "SELECT n FROM claim", tables_used: ["claim"], narrowed: [], preview: null,
      notes: [], rows_ms: 10 };
    const turns = [{ type: "RUN_STARTED", runId: "r1" }, { type: "CUSTOM", name: RESULT_EVENT, value: rows },
      { type: "CUSTOM", name: ANSWER_EVENT, value: SCALAR_ANSWER }].reduce(reduce, [] as Turn[]);
    expect(turns[0]!.result).toEqual(rows);
    expect(turns[0]!.answer).toEqual(SCALAR_ANSWER);
  });

  it("ignores a CUSTOM event that is not the answer payload", () => {
    const before = run(answered());
    const after = reduce(before, { type: "CUSTOM", name: "other.thing", value: 1 });
    expect(after).toBe(before);
  });

  // assistant-ui short-circuits message conversion on reference equality, so a
  // no-op event must not produce a new array.
  it("returns the same array reference when nothing changed", () => {
    const before = run(answered());
    expect(reduce(before, { type: "RUN_FINISHED", runId: "r1" })).toBe(before);
    expect(reduce(before, { type: "TEXT_MESSAGE_CONTENT", messageId: "m", delta: "" })).toBe(
      before,
    );
  });

  it("does not crash when a stray event arrives before any run started", () => {
    expect(reduce([], { type: "RUN_FINISHED", runId: "r1" })).toEqual([]);
    expect(reduce([], { type: "CUSTOM", name: ANSWER_EVENT, value: SCALAR_ANSWER })).toEqual(
      [],
    );
  });
});

describe("a run that ends without its own events", () => {
  it("an error with no run of its own leaves the previous answer complete and opens its own turn", () => {
    const first = run(answered());
    const turns = reduce(first, { type: "RUN_ERROR", message: "the engine returned 502" });
    expect(turns[0]!.status).toBe("complete");
    expect(turns.filter((t) => t.role === "assistant")).toHaveLength(2);
    expect(turns[1]!.status).toBe("error");
    expect(turns[1]!.error).toBe("the engine returned 502");
  });

  it("a failed request on a later question leaves the earlier answer as it was", async () => {
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async () => new Response("bad gateway", { status: 502, statusText: "Bad Gateway" })) as typeof fetch;
    try {
      let turns = [...run(answered()), userTurn("q2")];
      for await (const event of streamChat("q2", "thinking")) turns = reduce(turns, event);
      expect(turns[0]!.status).toBe("complete");
      expect(turns[0]!.text).toBe("There are 2 claims.");
      expect(turns.filter((t) => t.role === "assistant")).toHaveLength(2);
      expect(turns[1]!.text).toBe("q2");
      expect(turns.at(-1)!.status).toBe("error");
    } finally {
      globalThis.fetch = realFetch;
    }
  });

  it("an error turn of its own keeps the error's code", () => {
    const turns = reduce(run(answered()), { type: "RUN_ERROR", message: "down", code: "unavailable" });
    expect(turns.at(-1)!.errorCode).toBe("unavailable");
  });

  it("a settled turn says the connection ended, by its own code", () => {
    const settled = settle(run([{ type: "RUN_STARTED", runId: "r2" }]));
    expect(settled[0]!.errorCode).toBe(CONNECTION_ENDED_CODE);
  });

  it("a turn whose answer arrived and whose last frame was lost settles complete", () => {
    const turns = run(answered().slice(0, -1));
    expect(isRunning(turns)).toBe(true);
    const settled = settle(turns);
    expect(settled[0]!.status).toBe("complete");
    expect(settled[0]!.answer).toBeDefined();
  });

  it("a stream that carried no frame at all gives the question an error turn of its own", () => {
    const asked = [...run(answered()), userTurn("q2")];
    const closed = closeRun(asked, false);
    expect(closed).toHaveLength(asked.length + 1);
    expect(closed.at(-1)!.status).toBe("error");
    expect(closed.at(-1)!.error).toBe(CONNECTION_ENDED);
    expect(closed[0]!.status).toBe("complete");
  });

  it("a stream that ended its run is left as it is", () => {
    const turns = run(answered());
    expect(closeRun(turns, true)).toBe(turns);
  });

  it("settle turns a running turn into an error and leaves a finished one alone", () => {
    const turns = run([...answered(), { type: "RUN_STARTED", runId: "r2" }]);
    const settled = settle(turns);
    expect(settled[0]!.status).toBe("complete");
    expect(settled[1]!.status).toBe("error");
    expect(settled[1]!.error).toBe(CONNECTION_ENDED);
    expect(settle(turns, "cut")[1]!.error).toBe("cut");
  });

  it("settle returns the same array when nothing runs", () => {
    const turns = run(answered());
    expect(settle(turns)).toBe(turns);
  });
});

describe("isRunning", () => {
  it("is true only while a turn is still open", () => {
    expect(isRunning(run([{ type: "RUN_STARTED", runId: "r1" }]))).toBe(true);
    expect(isRunning(run(answered()))).toBe(false);
  });
});
