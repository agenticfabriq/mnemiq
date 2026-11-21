import { beforeEach, describe, expect, it, vi } from "vitest";

describe("the panel's entry", () => {
  beforeEach(() => vi.resetModules());

  it("registers one chat with its id", async () => {
    // The fake host as the entry sees it: imported after the reset, as the entry imports it.
    const { host } = await import("./test/core");
    host.reset();
    await import("./index");
    await import("./index");
    expect(host.registered).toHaveLength(1);
    const [{ chat, trigger, panel }] = host.registered;
    expect(chat).toEqual({ id: "agenticfabriq.mnemiq", name: "mnemiq",
      description: "Ask about your data and build dashboards" });
    expect(typeof trigger).toBe("function");
    expect(typeof panel).toBe("function");
  });
});
