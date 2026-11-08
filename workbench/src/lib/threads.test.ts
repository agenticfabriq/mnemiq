import { beforeEach, describe, expect, it } from "vitest";

import { SCALAR_ANSWER } from "../test/fixtures";
import { keyFor, SINGLE } from "./identity";
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

const NS = SINGLE;

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
    save(NS, [thread]);

    const [restored] = load(NS);
    expect(restored!.id).toBe(thread.id);
    expect(restored!.turns).toHaveLength(2);
    expect(restored!.turns[1]!.answer?.sql).toBe(SCALAR_ANSWER.sql);
  });

  it("does not persist a thread nothing was asked in", () => {
    save(NS, [emptyThread(), withTurns(userTurn("q"))]);
    expect(load(NS)).toHaveLength(1);
  });

  it("starts empty rather than throwing on unreadable history", () => {
    localStorage.setItem(keyFor(NS, "threads"), "{not json");
    expect(load(NS)).toEqual([]);
  });

  it("ignores entries that are not threads", () => {
    localStorage.setItem(keyFor(NS, "threads"), JSON.stringify([{ id: "x" }, null, 7]));
    expect(load(NS)).toEqual([]);
  });

  it("keeps at most 25 threads", () => {
    save(NS, Array.from({ length: 40 }, (_, i) => withTurns(userTurn(`q${i}`))));
    expect(load(NS)).toHaveLength(25);
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
      save(NS, [withTurns(userTurn("a")), withTurns(userTurn("b")), withTurns(userTurn("c"))]);
    } finally {
      Storage.prototype.setItem = real;
    }
    expect(load(NS)).toHaveLength(1);
  });
});

describe("per identity", () => {
  it("never reads one namespace's threads from another", () => {
    const thread = withTurns(userTurn("a's question"));
    save("ns-a", [thread]);

    expect(load("ns-b")).toEqual([]);
    expect(load("ns-a").map((t) => t.id)).toEqual([thread.id]);
  });

  it("no longer reads the legacy key: only a namespace is ever rendered", () => {
    localStorage.setItem("mnemiq.threads.v1", JSON.stringify([withTurns(userTurn("legacy"))]));

    expect(load(NS)).toEqual([]);
  });

  it("keeps nothing for a user who has no namespace", () => {
    save(null, [withTurns(userTurn("q"))]);

    expect(load(null)).toEqual([]);
    expect(localStorage.length).toBe(0);
  });
});

describe("isBlank", () => {
  it("is true only before anything is asked", () => {
    expect(isBlank(emptyThread())).toBe(true);
    expect(isBlank(withTurns(userTurn("q")))).toBe(false);
  });
});
