import React from "react";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { host } from "./test/core";
import { driver, resetBuilds } from "./build";
import { resetChanges } from "./change";
import { resetAsking } from "./ask";
import { getSnapshot, resetStore } from "./store";
import { resetDismissed } from "./views/Unfinished";

const BASE = "/extensions/agenticfabriq/mnemiq/v1";
const BUILD_ID = "0f6d0c1e-6b8f-4a39-9a4e-3d5f5a9c2b10";
const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

type Call = { method: string; path: string; body: unknown };
type Handler = (call: Call) => Response | Promise<Response>;

const DASHBOARDS: Record<string, Handler> = {
  "GET /api/v1/dashboard/1": () => json(200, { id: 1, result: { id: 1, dashboard_title: "One" } }),
  "GET /api/v1/dashboard/2": () => json(200, { id: 2, result: { id: 2, dashboard_title: "Two" } }),
  "GET /api/v1/dashboard/9": () => json(403, { message: "Forbidden" }),
};

const answer = (fields: object = {}) => ({ changed: [], left: [], dropped: [], not_editor: [], removed: [],
  would_remove: [], added: null, message: "", ...fields });

async function inPanel(path: string, handlers: Record<string, Handler>) {
  history.replaceState(null, "", path);
  host.navigate(path.startsWith("/dashboard/") ? "dashboard" : "chart_list");
  const calls: Call[] = [];
  const fetchFn = vi.fn<typeof fetch>(async (input, init) => {
    const url = String(input);
    const path = url.startsWith(BASE) ? url.slice(BASE.length) : url;
    const call = { method: init?.method ?? "GET", path, body: init?.body ? JSON.parse(String(init.body)) : null };
    calls.push(call);
    const routes: Record<string, Handler> = { "GET /schema": () => json(200, { tables: [], starters: [] }),
      "GET /superset/builds": () => json(200, { builds: [] }), ...DASHBOARDS, ...handlers };
    const handler = routes[`${call.method} ${path.split("?")[0]}`];
    return handler ? handler(call) : json(404, { detail: "Not Found" });
  });
  vi.stubGlobal("fetch", fetchFn);
  const { Panel } = await import("./Panel");
  render(<Panel />);
  await screen.findByRole("textbox");
  return calls;
}

function sendChange(request: string) {
  fireEvent.click(screen.getByText("Change", { exact: true }));
  fireEvent.change(screen.getByRole("textbox", { name: "Describe the change" }), { target: { value: request } });
  fireEvent.click(screen.getByRole("button", { name: "Change" }));
}

const changes = (calls: Call[]) => calls.filter((c) => c.method === "POST" && /\/changes$/.test(c.path));

beforeEach(() => {
  host.reset();
  resetStore();
  resetAsking();
  resetBuilds();
  resetChanges();
  resetDismissed();
});
afterEach(() => {
  driver.stop();
  cleanup();
  vi.unstubAllGlobals();
});

describe("change by chat", () => {
  it("a change goes to the dashboard on screen when it is sent", async () => {
    const calls = await inPanel("/dashboard/1/", { "POST /superset/dashboards/2/changes": () => json(200, answer({ message: "done" })) });
    await screen.findByRole("region", { name: "This dashboard" });
    // Superset moves to dashboard 2 with no page event, and the panel has not read the path again yet.
    history.pushState(null, "", "/dashboard/2/");
    sendChange("show daily instead of monthly");
    await vi.waitFor(() => expect(changes(calls)).toHaveLength(1));
    expect(changes(calls)[0]).toEqual({ method: "POST", path: "/superset/dashboards/2/changes",
      body: { request: "show daily instead of monthly" } });
  });

  it("Change is offered only on a readable dashboard", async () => {
    await inPanel("/chart/list/", {});
    expect(screen.queryByText("Change", { exact: true })).toBeNull();
    cleanup();
    await inPanel("/dashboard/9/", {});
    await new Promise((done) => setTimeout(done, 30));
    expect(screen.queryByText("Change", { exact: true })).toBeNull();
    cleanup();
    await inPanel("/dashboard/1/", {});
    await screen.findByRole("region", { name: "This dashboard" });
    expect(screen.getByText("Change", { exact: true })).toBeTruthy();
  });

  it("a change refused as not an editor names the editors", async () => {
    await inPanel("/dashboard/1/", { "POST /superset/dashboards/1/changes": () => json(200, answer({
      not_editor: [{ chart_id: 5, editors: ["Ana Admin", "Bo Owner"] }],
      message: "You are not an editor of a chart this change would set, so it was left." })) });
    await screen.findByRole("region", { name: "This dashboard" });
    sendChange("make the revenue chart a line");
    const card = await screen.findByRole("region", { name: "Change" });
    await within(card).findByText("You are not an editor of a chart this change would set, so it was left.");
    expect(card.textContent).toContain("Ana Admin, Bo Owner");
  });

  it("the answer lists what changed and what was left", async () => {
    await inPanel("/dashboard/1/", { "POST /superset/dashboards/1/changes": () => json(200, answer({
      changed: [101], left: [{ chart_id: 102, reason: "its type was set by hand, so it is kept" }],
      message: "1 chart changed, 1 left." })) });
    await screen.findByRole("region", { name: "This dashboard" });
    sendChange("show daily instead of monthly");
    const card = await screen.findByRole("region", { name: "Change" });
    await within(card).findByText("1 chart changed, 1 left.");
    expect(card.textContent).toMatch(/Changed: chart 101/);
    expect(card.textContent).toMatch(/Left: chart 102: its type was set by hand, so it is kept/);
    expect(within(card).getByRole("button", { name: "Reload to see the changes" })).toBeTruthy();
  });

  it("a removal is proposed, and only the confirmed charts are removed", async () => {
    const calls = await inPanel("/dashboard/1/", { "POST /superset/dashboards/1/changes": (call) =>
      (call.body as { remove_chart_ids?: number[] }).remove_chart_ids
        ? json(200, answer({ removed: [7], message: "Old KPI was deleted, moved to Superset's trash." }))
        : json(200, answer({ would_remove: [{ chart_id: 7, title: "Old KPI" }], message: "This would remove 1 chart." })) });
    await screen.findByRole("region", { name: "This dashboard" });
    sendChange("remove the old KPI");
    const card = await screen.findByRole("region", { name: "Change" });
    await within(card).findByText("This would remove 1 chart.");
    expect(card.textContent).toContain("Old KPI");
    // What confirming does, said before it is done: the charts are deleted.
    expect(card.textContent).toMatch(/This would delete: Old KPI/);
    expect(card.textContent).toMatch(/moved to Superset's trash/);
    expect(changes(calls)).toHaveLength(1);
    fireEvent.click(within(card).getByRole("button", { name: "Delete them" }));
    await within(screen.getByRole("region", { name: "Change" })).findByText("Old KPI was deleted, moved to Superset's trash.");
    expect(screen.getByRole("region", { name: "Change" }).textContent).toMatch(/Deleted: chart 7/);
    expect(changes(calls)[1].body).toEqual({ remove_chart_ids: [7] });
  });

  it("a chart the change adds runs as a build", async () => {
    let done = 0;
    const view = (n: number) => ({ build_id: BUILD_ID, status: n >= 3 ? "done" : "building", dashboard_id: 1,
      steps: ["dashboard", "funnel", "layout"].map((key, i) => ({ key, kind: key === "funnel" ? "chart" : key,
        status: i < n ? "done" : "pending", object_id: null, started_at: null, lease_until: null })),
      checks: {}, ran: null, message: null, check_reasons: {},
      plan: { charts: [{ key: "funnel", title: "Deals by stage", family: "funnel", measures: [] }], measures: [] } });
    await inPanel("/dashboard/1/", {
      "POST /superset/dashboards/1/changes": () => json(200, answer({ added: { build: view(0) }, message: "A funnel is being added." })),
      [`POST /superset/builds/${BUILD_ID}/next`]: () => json(200, { build: view(++done) }),
    });
    await screen.findByRole("region", { name: "This dashboard" });
    sendChange("add a funnel of deals by stage");
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
    expect(done).toBe(3);
    expect(getSnapshot().build?.goal).toBe("add a funnel of deals by stage");
  });

  it("an answer missing its lists reads them as none", async () => {
    await inPanel("/dashboard/1/", { "POST /superset/dashboards/1/changes": () => json(200, { message: "Nothing to change." }) });
    await screen.findByRole("region", { name: "This dashboard" });
    sendChange("make it nicer");
    const card = await screen.findByRole("region", { name: "Change" });
    await within(card).findByText("Nothing to change.");
    expect(card.textContent).not.toMatch(/Changed:|Left:/);
  });

  it("a proposal made on one dashboard cannot be confirmed from another", async () => {
    const calls = await inPanel("/dashboard/1/", { "POST /superset/dashboards/1/changes": () =>
      json(200, answer({ would_remove: [{ chart_id: 7, title: "Revenue" }], message: "This would remove 1 chart." })) });
    await screen.findByRole("region", { name: "This dashboard" });
    sendChange("remove the revenue chart");
    await within(await screen.findByRole("region", { name: "Change" })).findByText("This would remove 1 chart.");
    history.pushState(null, "", "/dashboard/2/");
    window.dispatchEvent(new PopStateEvent("popstate"));
    await vi.waitFor(() => expect(screen.getByRole("region", { name: "This dashboard" }).textContent).toContain("Two"));
    const card = screen.getByRole("region", { name: "Change" });
    expect(within(card).queryByRole("button", { name: "Delete them" })).toBeNull();
    expect(within(card).queryByRole("button", { name: "Reload to see the changes" })).toBeNull();
    expect(card.textContent).toMatch(/made on another dashboard/);
    expect(changes(calls)).toHaveLength(1);
  });

  it("entries of a shape the panel does not read are left out, never a crash", async () => {
    await inPanel("/dashboard/1/", { "POST /superset/dashboards/1/changes": () => json(200, {
      changed: [101, "x", null], left: [null, { chart_id: "9", reason: 4 }, { chart_id: 102, reason: "kept by hand" }],
      dropped: [{}], not_editor: [{ chart_id: 1, editors: "Ana" }, { chart_id: 5, editors: ["Bo", 3] }],
      removed: null, would_remove: [{ chart_id: null }], added: { offer: null }, message: "Partly read." }) });
    await screen.findByRole("region", { name: "This dashboard" });
    sendChange("show daily instead of monthly");
    const card = await screen.findByRole("region", { name: "Change" });
    await within(card).findByText("Partly read.");
    expect(card.textContent).toMatch(/Changed: chart 101$|Changed: chart 101[^,]/m);
    expect(card.textContent).toMatch(/Left: chart 102: kept by hand/);
    expect(card.textContent).toMatch(/You do not edit chart 5; its editors: Bo/);
    expect(card.textContent).not.toMatch(/chart 9|chart 1;|would delete/);
  });

  it("a change mnemiq refuses because the person does not edit the dashboard says so, with the editors", async () => {
    const sentence = "You are not an editor of this dashboard; its editors are Ana Admin and Bo Owner.";
    await inPanel("/dashboard/1/", { "POST /superset/dashboards/1/changes": () => json(403, { detail: sentence,
      end_session_url: null }) });
    await screen.findByRole("region", { name: "This dashboard" });
    sendChange("show daily instead of monthly");
    const card = await screen.findByRole("region", { name: "Change" });
    await within(card).findByText(sentence);
  });
});
