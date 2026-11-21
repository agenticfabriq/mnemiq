import React from "react";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { host } from "./test/core";
import { PanelApi } from "./api";
import { probe } from "./readiness";
import { Panel } from "./Panel";
import { resetStore } from "./store";

const NOT_A_BUILDER = "Superset refused this user's queries. Building with mnemiq needs Superset's Alpha and sql_lab roles.";
const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

describe("probe", () => {
  const api = (answer: () => Response | Promise<Response>) => new PanelApi("", async () => "t", vi.fn(async () => answer()));

  it("is ready with mnemiq's starters", async () => {
    expect(await probe(api(() => json(200, { tables: [{ object_id: "x" }], starters: ["How many policies?"] }))))
      .toEqual({ kind: "ready", starters: ["How many policies?"] });
  });

  it("names a refusal by its kind, with its sentence", async () => {
    expect(await probe(api(() => json(403, { detail: NOT_A_BUILDER }))))
      .toEqual({ kind: "not-a-builder", sentence: NOT_A_BUILDER });
    expect(await probe(api(() => json(401, {})))).toMatchObject({ kind: "signed-out" });
    expect(await probe(api(() => json(504, { detail: "mnemiq took too long" }))))
      .toEqual({ kind: "unavailable", sentence: "mnemiq took too long" });
    expect(await probe(api(() => json(200, { starters: "not a list" })))).toMatchObject({ kind: "unavailable" });
  });
});

describe("the panel's readiness", () => {
  beforeEach(() => {
    host.reset();
    resetStore();
    history.replaceState(null, "", "/dashboard/7/");
    host.navigate("dashboard");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("a non-builder sees the roles sentence and nothing else", async () => {
    const fetchFn = vi.fn(async () => json(403, { detail: NOT_A_BUILDER }));
    vi.stubGlobal("fetch", fetchFn);
    render(<Panel />);
    await screen.findByText(NOT_A_BUILDER);
    // Give any further request its chance before counting.
    await new Promise((done) => setTimeout(done, 50));
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect((fetchFn.mock.calls[0] as unknown as [string])[0]).toBe("/extensions/agenticfabriq/mnemiq/v1/schema");
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.queryByRole("region", { name: "Starters" })).toBeNull();
    expect(screen.queryByRole("region", { name: "This dashboard" })).toBeNull();
    expect(screen.queryByText(/measure/i)).toBeNull();
    expect(screen.queryByRole("status", { name: "Unfinished build" })).toBeNull();
  });

  it("a builder sees the starters and the dashboard on screen", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => url.includes("/api/v1/dashboard/")
      ? json(200, { id: 7, result: { id: 7, dashboard_title: "Claims" } })
      : json(200, { tables: [], starters: ["How many claims are open?"] })));
    render(<Panel />);
    expect(await screen.findByText("How many claims are open?")).toBeTruthy();
    const dashboard = await screen.findByRole("region", { name: "This dashboard" });
    expect(dashboard.textContent).toContain("Claims");
    expect(screen.getByRole("region", { name: "Measures" })).toBeTruthy();
  });

  it("a dashboard the person cannot read gives no dashboard actions", async () => {
    history.replaceState(null, "", "/dashboard/secret-board/");
    vi.stubGlobal("fetch", vi.fn(async (url: string) => url.includes("/api/v1/dashboard/")
      ? json(403, { message: "Forbidden" })
      : json(200, { tables: [], starters: ["How many claims are open?"] })));
    render(<Panel />);
    await screen.findByText("How many claims are open?");
    await new Promise((done) => setTimeout(done, 50));
    expect(screen.queryByRole("region", { name: "This dashboard" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Measures" })).toBeNull();
    expect(document.body.textContent).not.toContain("secret-board");
  });

  it("reopening asks again before showing anything", async () => {
    let refuse = false;
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const fetchFn = vi.fn(async (url: string) => {
      if (url.includes("/api/v1/dashboard/")) return json(200, { id: 7, result: { id: 7, dashboard_title: "Claims" } });
      if (url.includes("/superset/builds")) return json(200, { builds: [] });
      if (refuse) {
        await held;
        return json(403, { detail: NOT_A_BUILDER });
      }
      return json(200, { tables: [], starters: ["How many claims are open?"] });
    });
    vi.stubGlobal("fetch", fetchFn);
    const first = render(<Panel />);
    await screen.findByRole("textbox");
    first.unmount();
    // The person's roles changed while the panel was closed.
    refuse = true;
    fetchFn.mockClear();
    render(<Panel />);
    await new Promise((done) => setTimeout(done, 50));
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(fetchFn.mock.calls.map((call) => (call as unknown as [string])[0]))
      .toEqual(["/extensions/agenticfabriq/mnemiq/v1/schema"]);
    release();
    await screen.findByText(NOT_A_BUILDER);
    expect(screen.queryByRole("textbox")).toBeNull();
  });
});
