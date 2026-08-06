import { describe, expect, it } from "vitest";

import { SCALAR_ANSWER, TABLE_ANSWER, DEFERRAL } from "../test/fixtures";
import { historyFrom } from "./history";
import { userTurn, type Turn } from "./store";
import type { AnswerPayload } from "./types";

const answered = (text: string, answer: AnswerPayload): Turn => ({
  id: `a-${text}`,
  role: "assistant",
  text,
  answer,
  status: "complete",
});

const stamped = (answer: AnswerPayload, fp = "fp-analyst"): AnswerPayload => ({
  ...answer,
  grant_fingerprint: fp,
});

describe("historyFrom", () => {
  it("pairs each answer with the question it replied to", () => {
    const turns = [
      userTurn("which region has the most?"),
      answered("The west region.", stamped(TABLE_ANSWER)),
    ];

    const history = historyFrom(turns);

    expect(history).toHaveLength(1);
    expect(history[0]!.question).toBe("which region has the most?");
    expect(history[0]!.sql).toBe(TABLE_ANSWER.sql);
  });

  it("carries the result rows, which is what a pronoun resolves against", () => {
    const turns = [userTurn("q"), answered("a", stamped(TABLE_ANSWER))];

    const history = historyFrom(turns);

    expect(history[0]!.columns).toEqual(TABLE_ANSWER.preview!.columns);
    expect(history[0]!.rows.length).toBeGreaterThan(0);
  });

  it("echoes the boundary the engine stamped, so the engine can re-check it", () => {
    const turns = [userTurn("q"), answered("a", stamped(SCALAR_ANSWER, "fp-claims-lead"))];

    expect(historyFrom(turns)[0]!.grant_fingerprint).toBe("fp-claims-lead");
  });

  it("leaves out a deferral: it has no result to resolve against", () => {
    const turns = [userTurn("q"), answered("declined", stamped(DEFERRAL))];

    expect(historyFrom(turns)).toEqual([]);
  });

  it("leaves out a failure for the same reason", () => {
    const failed = stamped({ ...SCALAR_ANSWER, failed: true });
    expect(historyFrom([userTurn("q"), answered("outage", failed)])).toEqual([]);
  });

  it("leaves out a turn still running", () => {
    const running: Turn = { id: "a", role: "assistant", text: "", status: "running" };
    expect(historyFrom([userTurn("q"), running])).toEqual([]);
  });

  it("keeps only the last few turns, newest last", () => {
    const turns: Turn[] = [];
    for (let i = 0; i < 6; i++) {
      turns.push(userTurn(`q${i}`), answered(`a${i}`, stamped(SCALAR_ANSWER)));
    }

    const history = historyFrom(turns);

    expect(history).toHaveLength(3);
    expect(history[2]!.question).toBe("q5");
  });

  it("bounds the rows it sends", () => {
    const big: AnswerPayload = stamped({
      ...TABLE_ANSWER,
      preview: {
        columns: ["n"],
        rows: Array.from({ length: 40 }, (_, i) => [i]),
        row_count: 40,
        truncated: false,
      },
    });

    expect(historyFrom([userTurn("q"), answered("a", big)])[0]!.rows).toHaveLength(5);
  });

  it("is empty for a fresh thread", () => {
    expect(historyFrom([])).toEqual([]);
  });
});
