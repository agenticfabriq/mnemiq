import { beforeEach, describe, expect, it } from "vitest";

import { SCALAR_ANSWER } from "../test/fixtures";
import { userTurn, type Turn } from "./store";
import { emptyThread, isBlank, load, save, titleFor, uid } from "./threads";

const assistantTurn = (): Turn => ({
  id: uid(),
  role: "assistant",
  text: SCALAR_ANSWER.answer,
  answer: SCALAR_ANSWER,
  status: "complete",
});

const withTurns = (...turns: Turn[]) => ({ ...emptyThread(), turns });

beforeEach(() => localStorage.clear());

describe("titleFor", () => {
  it("names a thread by the first question asked", () => {
    const thread = withTurns(userTurn("how many claims are there?"), assistantTurn());
    expect(titleFor(thread)).toBe("how many claims are there?");
  });

  it("keeps a long question to one line", () => {
    const long = "list every fire claim with its identifier, its amount, and the party";
    const title = titleFor(withTurns(userTurn(long)));
    expect(title).toHaveLength(48);
    expect(title.endsWith("…")).toBe(true);
  });

  it("collapses newlines rather than breaking the row", () => {
    expect(titleFor(withTurns(userTurn("how many\n\n  claims?")))).toBe("how many claims?");
  });

  it("falls back when nothing has been asked yet", () => {
    expect(titleFor(emptyThread())).toBe("New chat");
  });
});

describe("persistence", () => {
  it("round-trips a thread with its answer payload intact", () => {
    const thread = withTurns(userTurn("how many claims are there?"), assistantTurn());
    save([thread]);

    const [restored] = load();
    expect(restored!.id).toBe(thread.id);
    expect(restored!.turns).toHaveLength(2);
    expect(restored!.turns[1]!.answer?.sql).toBe(SCALAR_ANSWER.sql);
  });

  it("does not persist a thread nothing was asked in", () => {
    save([emptyThread(), withTurns(userTurn("q"))]);
    expect(load()).toHaveLength(1);
  });

  it("starts empty rather than throwing on unreadable history", () => {
    localStorage.setItem("mnemiq.threads.v1", "{not json");
    expect(load()).toEqual([]);
  });

  it("ignores entries that are not threads", () => {
    localStorage.setItem("mnemiq.threads.v1", JSON.stringify([{ id: "x" }, null, 7]));
    expect(load()).toEqual([]);
  });

  it("keeps at most 25 threads", () => {
    save(Array.from({ length: 40 }, (_, i) => withTurns(userTurn(`q${i}`))));
    expect(load()).toHaveLength(25);
  });

  it("drops the oldest rather than losing everything when storage is full", () => {
    const real = Storage.prototype.setItem;
    let calls = 0;
    Storage.prototype.setItem = function (key: string, value: string) {
      // Refuse the first two writes, as a quota error would.
      if (calls++ < 2) throw new DOMException("quota", "QuotaExceededError");
      return real.call(this, key, value);
    };
    try {
      save([withTurns(userTurn("a")), withTurns(userTurn("b")), withTurns(userTurn("c"))]);
    } finally {
      Storage.prototype.setItem = real;
    }
    expect(load()).toHaveLength(1);
  });
});

describe("isBlank", () => {
  it("is true only before anything is asked", () => {
    expect(isBlank(emptyThread())).toBe(true);
    expect(isBlank(withTurns(userTurn("q")))).toBe(false);
  });
});
