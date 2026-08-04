import { describe, expect, it } from "vitest";

import { DEFERRAL, SCALAR_ANSWER } from "../test/fixtures";
import { ANSWER_EVENT, isRunning, reduce, userTurn, type Turn } from "./store";
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

describe("isRunning", () => {
  it("is true only while a turn is still open", () => {
    expect(isRunning(run([{ type: "RUN_STARTED", runId: "r1" }]))).toBe(true);
    expect(isRunning(run(answered()))).toBe(false);
  });
});
