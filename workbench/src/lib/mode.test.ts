import { beforeEach, describe, expect, it } from "vitest";

import { keyFor, SINGLE } from "./identity";
import { loadMode, saveMode } from "./mode";
import { DEFAULT_MODE, MODES } from "./types";

const NS = SINGLE;

beforeEach(() => localStorage.clear());

describe("mode persistence", () => {
  it("remembers the mode across a reload", () => {
    // The bug, verbatim: pick `deep`, refresh, and the workbench is back on `thinking`.
    // Mode was plain component state, so every reload discarded the choice and silently
    // changed what the next question would cost.
    saveMode(NS, "deep");

    expect(loadMode(NS)).toBe("deep");
  });

  it("starts on the default when nothing was chosen", () => {
    expect(loadMode(NS)).toBe(DEFAULT_MODE);
  });

  it("refuses a stored value that is not a mode", () => {
    // Storage is the user's to edit and survives a release that renames a mode. An
    // unrecognised value must fall back rather than reach the engine as a mode it has
    // never heard of -- the engine would refuse the question, and the reason would look
    // like the question's fault.
    localStorage.setItem(keyFor(NS, "mode"), "turbo");

    expect(loadMode(NS)).toBe(DEFAULT_MODE);
  });

  it("survives storage that cannot be read at all", () => {
    // `threads.ts` treats unreadable history as "no history" rather than a failure to
    // start; the same rule applies to a preference.
    localStorage.setItem(keyFor(NS, "mode"), "");

    expect(loadMode(NS)).toBe(DEFAULT_MODE);
  });

  it("round-trips every mode the engine offers", () => {
    // Non-vacuity: a `loadMode` that always answered `thinking` would pass the first
    // test above by accident if the default ever changed to `deep`.
    for (const mode of MODES) {
      saveMode(NS, mode);
      expect(loadMode(NS)).toBe(mode);
    }
  });

  it("remembers each identity's mode apart", () => {
    saveMode("ns-a", "deep");

    expect(loadMode("ns-b")).toBe(DEFAULT_MODE);
    expect(loadMode("ns-a")).toBe("deep");
  });

  it("keeps no mode for a user who has no namespace", () => {
    saveMode(null, "deep");

    expect(loadMode(null)).toBe(DEFAULT_MODE);
    expect(localStorage.length).toBe(0);
  });
});
