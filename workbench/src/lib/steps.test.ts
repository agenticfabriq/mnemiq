import { describe, expect, it } from "vitest";

import { reduce, type Turn } from "./store";
import type { AguiEvent } from "./types";

const run = (events: AguiEvent[], from: Turn[] = []) => events.reduce(reduce, from);

const started = (id: string, name: string, extra: object = {}): AguiEvent =>
  ({ type: "STEP_STARTED", stepId: id, stepName: name, ...extra }) as AguiEvent;

const finished = (id: string, name: string, ms: number, ok = true): AguiEvent =>
  ({ type: "STEP_FINISHED", stepId: id, stepName: name, durationMs: ms, ok }) as AguiEvent;

describe("step reduction", () => {
  it("opens a step as running and closes it with its duration", () => {
    const turns = run([
      { type: "RUN_STARTED", runId: "r1" },
      started("s1", "retrieve"),
      finished("s1", "retrieve", 1866),
    ]);

    expect(turns[0]!.steps).toEqual([
      { id: "s1", name: "retrieve", status: "done", ms: 1866 },
    ]);
  });

  it("keeps a step running until its finish arrives", () => {
    const turns = run([{ type: "RUN_STARTED", runId: "r1" }, started("s1", "plan")]);
    expect(turns[0]!.steps![0]).toMatchObject({ status: "running" });
    expect(turns[0]!.steps![0]!.ms).toBeUndefined();
  });

  it("records the phases in the order the engine reported them", () => {
    const turns = run([
      { type: "RUN_STARTED", runId: "r1" },
      started("s1", "retrieve"),
      finished("s1", "retrieve", 1866),
      started("s2", "plan", { attempt: 1, of: 3 }),
      finished("s2", "plan", 11559),
      started("s3", "execute"),
      finished("s3", "execute", 17),
    ]);

    expect(turns[0]!.steps!.map((s) => s.name)).toEqual(["retrieve", "plan", "execute"]);
    expect(turns[0]!.steps![1]).toMatchObject({ attempt: 1, of: 3, ms: 11559 });
  });

  // The whole reason the wire carries an id: deep mode's candidates share a name.
  it("closes the right step when two of the same name overlap", () => {
    const turns = run([
      { type: "RUN_STARTED", runId: "r1" },
      started("a", "candidate", { index: 1, of: 5 }),
      started("b", "candidate", { index: 2, of: 5 }),
      finished("b", "candidate", 3717),
    ]);

    const [first, second] = turns[0]!.steps!;
    expect(first).toMatchObject({ index: 1, status: "running" });
    expect(second).toMatchObject({ index: 2, status: "done", ms: 3717 });
  });

  it("marks a step that raised as failed", () => {
    const turns = run([
      { type: "RUN_STARTED", runId: "r1" },
      started("s1", "execute"),
      finished("s1", "execute", 12, false),
    ]);
    expect(turns[0]!.steps![0]!.status).toBe("failed");
  });

  it("ignores a finish for a step it never saw start", () => {
    const before = run([{ type: "RUN_STARTED", runId: "r1" }, started("s1", "plan")]);
    const after = reduce(before, finished("ghost", "plan", 5));
    expect(after).toBe(before);
  });

  it("ignores a malformed step event rather than pushing a nameless row", () => {
    const before = run([{ type: "RUN_STARTED", runId: "r1" }]);
    expect(reduce(before, { type: "STEP_STARTED" } as AguiEvent)).toBe(before);
  });

  it("leaves turns without steps undisturbed", () => {
    const turns = run([{ type: "RUN_STARTED", runId: "r1" }]);
    expect(turns[0]!.steps).toBeUndefined();
  });
});
