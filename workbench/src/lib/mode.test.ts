import { beforeEach, describe, expect, it } from "vitest";

import { loadMode, saveMode } from "./mode";
import { DEFAULT_MODE, MODES } from "./types";

beforeEach(() => localStorage.clear());

describe("mode persistence", () => {
  it("remembers the mode across a reload", () => {
    // The bug, verbatim: pick `deep`, refresh, and the workbench is back on `thinking`.
    // Mode was plain component state, so every reload discarded the choice and silently
    // changed what the next question would cost.
    saveMode("deep");

    expect(loadMode()).toBe("deep");
  });

  it("starts on the default when nothing was chosen", () => {
    expect(loadMode()).toBe(DEFAULT_MODE);
  });

  it("refuses a stored value that is not a mode", () => {
    // Storage is the user's to edit and survives a release that renames a mode. An
    // unrecognised value must fall back rather than reach the engine as a mode it has
    // never heard of -- the engine would refuse the question, and the reason would look
    // like the question's fault.
    localStorage.setItem("mnemiq.mode.v1", "turbo");

    expect(loadMode()).toBe(DEFAULT_MODE);
  });

  it("survives storage that cannot be read at all", () => {
    // `threads.ts` treats unreadable history as "no history" rather than a failure to
    // start; the same rule applies to a preference.
    localStorage.setItem("mnemiq.mode.v1", "");

    expect(loadMode()).toBe(DEFAULT_MODE);
  });

  it("round-trips every mode the engine offers", () => {
    // Non-vacuity: a `loadMode` that always answered `thinking` would pass the first
    // test above by accident if the default ever changed to `deep`.
    for (const mode of MODES) {
      saveMode(mode);
      expect(loadMode()).toBe(mode);
    }
  });
});
