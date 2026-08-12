import { describe, expect, it } from "vitest";

import { effort } from "./verdict";

describe("effort — what the mode actually spent (M33)", () => {
  it("says plainly when nothing extra was needed", () => {
    expect(effort({ attempts: 1, corrected: false })).toBe("1 attempt · no repair needed");
  });

  it("names the corrector when it carried the plan", () => {
    expect(effort({ attempts: 1, corrected: true })).toBe("1 attempt · SQL corrected once");
  });

  it("pluralises a retried run", () => {
    expect(effort({ attempts: 2, corrected: false })).toBe("2 attempts · no repair needed");
  });

  it("reports nothing rather than a fabricated zero when the engine reported neither", () => {
    expect(effort({ attempts: null, corrected: null })).toBeNull();
  });
});
