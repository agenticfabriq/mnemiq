import React from "react";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { host } from "./test/core";
import { PanelApi } from "./api";
import { BuildDriver, CHECK_WORDS, driver, resetBuilds, type BuildView, type PlanSummary } from "./build";
import { BuildCard } from "./views/BuildCard";
import { getSnapshot, resetStore, update } from "./store";
import { resetAsking } from "./ask";
import { resetDismissed } from "./views/Unfinished";

const BASE = "/extensions/agenticfabriq/mnemiq/v1";
const BUILD_ID = "0f6d0c1e-6b8f-4a39-9a4e-3d5f5a9c2b10";
const PLAN: PlanSummary = {
  charts: [
    { key: "open_claims", title: "Open claims", family: "big_number", measures: ["open_claims"] },
    { key: "claims_by_region", title: "Claims by region", family: "bar", measures: ["open_claims"] },
  ],
  measures: [{ name: "open_claims", label: "Open claims", expression: "COUNT(*)", kind: "count", currency: null }],
};
const KEYS: [string, string][] = [["dashboard", "dashboard"], ["open_claims", "chart"], ["claims_by_region", "chart"],
  ["layout", "layout"], ["check", "check"]];
const IDS: Record<string, number> = { dashboard: 7, open_claims: 101, claims_by_region: 102 };

/** The build after `done` of its five steps. */
function view(done: number, fields: Partial<BuildView> = {}): BuildView {
  return {
    build_id: BUILD_ID, status: done >= KEYS.length ? "done" : "building", dashboard_id: done > 0 ? 7 : null,
    steps: KEYS.map(([key, kind], i) => ({ key, kind, status: i < done ? "done" : "pending",
      object_id: i < done ? (IDS[key] ?? null) : null, started_at: null, lease_until: null })),
    checks: done >= KEYS.length ? { "101": "checked", "102": "not_checked" } : {},
    ran: done > 0 ? KEYS[done - 1][0] : null, message: null, plan: PLAN,
    check_reasons: done >= KEYS.length ? { "102": "Superset's answer for this chart could not be read" } : {},
    ...fields,
  };
}

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

type Call = { method: string; path: string; body: unknown };
type Handler = (call: Call) => Response | Promise<Response>;

/** mnemiq behind the forwarder, and Superset's dashboard API, answering by method and path; each call recorded,
 * and the most requests ever in flight at once. */
function server(handlers: Record<string, Handler>) {
  const calls: Call[] = [];
  let inFlight = 0;
  const seen = { most: 0 };
  const fetchFn = vi.fn<typeof fetch>(async (input, init) => {
    const url = String(input);
    const path = url.startsWith(BASE) ? url.slice(BASE.length) : url;
    const call = { method: init?.method ?? "GET", path, body: init?.body ? JSON.parse(String(init.body)) : null };
    calls.push(call);
    inFlight += 1;
    seen.most = Math.max(seen.most, inFlight);
    try {
      const handler = handlers[`${call.method} ${path.split("?")[0]}`];
      if (!handler) return json(404, { detail: "Not Found" });
      return await handler(call);
    } finally {
      inFlight -= 1;
    }
  });
  return { fetchFn, calls, seen };
}

/** A build of five steps answered one `next` at a time, from `from` steps done. */
function stepping(from = 0) {
  let done = from;
  return () => json(200, { build: view(++done) });
}

const api = (fetchFn: typeof fetch) => new PanelApi("", async () => "csrf", fetchFn);
const nexts = (calls: Call[]) => calls.filter((c) => c.path === `/superset/builds/${BUILD_ID}/next`);

beforeEach(() => {
  host.reset();
  resetStore();
  resetAsking();
  resetBuilds();
  resetDismissed();
  history.replaceState(null, "", "/chart/list/");
});
afterEach(async () => {
  driver.stop();
  cleanup();
  vi.unstubAllGlobals();
});

describe("the build driver", () => {
  it("a build runs one step per request until it is done", async () => {
    const { fetchFn, calls, seen } = server({
      "POST /superset/builds": () => json(201, { build: view(0) }),
      [`POST /superset/builds/${BUILD_ID}/next`]: stepping(),
    });
    const building = new BuildDriver();
    building.attach(api(fetchFn));
    const started = await building.start("open claims, by region", {});
    expect("build" in started).toBe(true);
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
    expect(nexts(calls)).toHaveLength(5);
    expect(seen.most).toBe(1);
    expect(calls[0]).toEqual({ method: "POST", path: "/superset/builds", body: { goal: "open claims, by region" } });
    expect(building.running()).toBeNull();
  });

  it("a busy step waits and tries again, then stops after five", async () => {
    const waits: number[] = [];
    const { fetchFn, calls, seen } = server({
      [`POST /superset/builds/${BUILD_ID}/next`]: () => json(409, { detail: "another request is running this build's step" }),
    });
    const building = new BuildDriver({ sleep: async (ms) => { waits.push(ms); } });
    building.attach(api(fetchFn));
    // Asked twice at once, it still runs one loop: never two steps at a time.
    await Promise.all([building.run(BUILD_ID), building.run(BUILD_ID)]);
    expect(waits).toEqual([2000, 4000, 8000, 16000, 16000]);
    expect(nexts(calls)).toHaveLength(6);
    expect(seen.most).toBe(1);
    expect(getSnapshot().build?.stopped).toMatch(/another request is running this build's step/i);
    expect(building.running()).toBeNull();
  });

  it("a forwarded step that times out leaves the build resumable", async () => {
    history.replaceState(null, "", "/dashboard/7/");
    host.navigate("dashboard");
    let timeOut = true;
    const { fetchFn, calls } = server({
      "GET /schema": () => json(200, { tables: [], starters: [] }),
      "GET /api/v1/dashboard/7": () => json(200, { id: 7, result: { id: 7, dashboard_title: "Claims" } }),
      "POST /superset/builds": () => json(201, { build: view(0, { dashboard_id: 7 }) }),
      [`POST /superset/builds/${BUILD_ID}/next`]: (() => {
        const next = stepping();
        return () => (timeOut && nexts(calls).length === 2
          ? json(504, { detail: "mnemiq took too long to answer" }) : next());
      })(),
      // Listed only once it has started: what the banner shows after the stop, it read after the stop.
      "GET /superset/builds": () => json(200, { builds: nexts(calls).length === 0 ? [] : [{ build_id: BUILD_ID,
        goal: "open claims, by region", dashboard_id: 7, status: "building", created_at: "2026-11-20T10:00:00Z",
        updated_at: "2026-11-20T10:01:00Z", steps_done: 1, steps_total: 5 }] }),
    });
    vi.stubGlobal("fetch", fetchFn);
    const { Panel } = await import("./Panel");
    render(<Panel />);
    await screen.findByRole("textbox");
    await act(async () => { await driver.start("open claims, by region", { dashboardId: 7 }); });
    await vi.waitFor(() => expect(getSnapshot().build?.stopped).toBe("mnemiq took too long to answer"));
    expect(driver.running()).toBeNull();
    const banner = await screen.findByRole("status", { name: "Unfinished build" });
    expect(banner.textContent).toContain("'open claims, by region' stopped after 1 of 5 steps");
    timeOut = false;
    fireEvent.click(within(banner).getByRole("button", { name: "Finish" }));
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
  });
});

describe("starting a build", () => {
  it("the plan is shown before the first chart is made", async () => {
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const next = stepping();
    const { fetchFn } = server({
      "POST /superset/builds": () => json(201, { build: view(0) }),
      [`POST /superset/builds/${BUILD_ID}/next`]: async () => { await held; return next(); },
    });
    driver.attach(api(fetchFn));
    await driver.start("open claims, by region", {});
    render(<BuildCard />);
    const plan = screen.getByRole("region", { name: "Plan" });
    expect(plan.textContent).toContain("Open claims");
    expect(plan.textContent).toContain("Claims by region");
    expect(plan.textContent).toContain("bar");
    expect(screen.getByRole("list", { name: "Steps" }).textContent).not.toMatch(/Open claims: done/);
    release();
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
  });

  it("an offer's three answers send what each says", async () => {
    const offer = { offer: { dashboard_id: 12, title: "Claims overview", reason: "it shows open claims",
      message: "Dashboard 12 already covers this goal, so nothing was built." } };
    const { fetchFn, calls } = server({
      "POST /superset/builds": (call) => (call.body as { dashboard_id?: number; fresh?: boolean }).dashboard_id
        || (call.body as { fresh?: boolean }).fresh ? json(201, { build: view(0) }) : json(200, offer),
      [`POST /superset/builds/${BUILD_ID}/next`]: stepping(),
    });
    driver.attach(api(fetchFn));
    await driver.start("open claims", {});
    render(<BuildCard />);
    expect(screen.getByRole("link", { name: "Open it" }).getAttribute("href")).toBe("/dashboard/12/");
    fireEvent.click(screen.getByRole("button", { name: "Add these charts to it" }));
    await vi.waitFor(() => expect(calls.filter((c) => c.path === "/superset/builds")).toHaveLength(2));
    expect(calls.filter((c) => c.path === "/superset/builds")[1].body).toEqual({ goal: "open claims", dashboard_id: 12 });
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
    cleanup();
    await driver.start("open claims", {});
    render(<BuildCard />);
    fireEvent.click(screen.getByRole("button", { name: "Build a new one" }));
    await vi.waitFor(() => expect(calls.filter((c) => c.path === "/superset/builds")).toHaveLength(4));
    expect(calls.filter((c) => c.path === "/superset/builds")[3].body).toEqual({ goal: "open claims", fresh: true });
  });

  it("a refused plan names each part", async () => {
    const { fetchFn } = server({
      "POST /superset/builds": () => json(200, { refused: { reason: "two parts of the goal could not be planned",
        parts: [{ part: "churn by month", reason: "no dataset holds churn" },
                { part: "claims by adjuster", reason: "no column names an adjuster" }] } }),
    });
    driver.attach(api(fetchFn));
    await driver.start("churn by month and claims by adjuster", {});
    render(<BuildCard />);
    const card = screen.getByRole("region", { name: "Build" });
    expect(card.textContent).toContain("two parts of the goal could not be planned");
    expect(within(card).getAllByRole("listitem").map((li) => li.textContent)).toEqual([
      "churn by month: no dataset holds churn", "claims by adjuster: no column names an adjuster"]);
  });
});

describe("the build card", () => {
  it("only a passed chart reads \"checked\"", () => {
    const statuses = Object.keys(CHECK_WORDS);
    expect(statuses).toEqual(["checked", "mismatch", "empty", "cut", "format_wrong", "not_checked"]);
    const charts = statuses.map((status, i) => ({ key: `c${i}`, title: `Chart ${i}`, family: "bar", measures: [] }));
    const done: BuildView = {
      build_id: BUILD_ID, status: "done", dashboard_id: 7, ran: "check", message: null,
      plan: { charts, measures: [] },
      steps: charts.map((chart, i) => ({ key: chart.key, kind: "chart", status: "done", object_id: 200 + i,
        started_at: null, lease_until: null })),
      checks: Object.fromEntries(statuses.map((status, i) => [String(200 + i), status])),
      check_reasons: {},
    };
    resetStore();
    act(() => driver.show("six charts", {}, done));
    render(<BuildCard />);
    const card = screen.getByRole("region", { name: "Build" });
    expect([...card.querySelectorAll("*")].filter((el) => el.children.length === 0 && el.textContent === "checked"))
      .toHaveLength(1);
    for (const words of Object.values(CHECK_WORDS)) expect(card.textContent).toContain(words);
  });
});

describe("finishing and the page", () => {
  it("an unfinished build of this dashboard is offered and Finish resumes it", async () => {
    history.replaceState(null, "", "/dashboard/7/");
    host.navigate("dashboard");
    const { fetchFn, calls } = server({
      "GET /schema": () => json(200, { tables: [], starters: [] }),
      "GET /api/v1/dashboard/7": () => json(200, { id: 7, result: { id: 7, dashboard_title: "Claims" } }),
      "GET /superset/builds": () => json(200, { builds: [{ build_id: BUILD_ID, goal: "claims by region",
        dashboard_id: 7, status: "building", created_at: "x", updated_at: "y", steps_done: 2, steps_total: 5 }] }),
      [`POST /superset/builds/${BUILD_ID}/next`]: stepping(2),
    });
    vi.stubGlobal("fetch", fetchFn);
    const { Panel } = await import("./Panel");
    render(<Panel />);
    const banner = await screen.findByRole("status", { name: "Unfinished build" });
    expect(banner.textContent).toContain("'claims by region' stopped after 2 of 5 steps");
    expect(calls.find((c) => c.path.startsWith("/superset/builds?"))?.path).toBe("/superset/builds?dashboard_id=7");
    fireEvent.click(within(banner).getByRole("button", { name: "Finish" }));
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
    expect(nexts(calls)).toHaveLength(3);
    expect(await screen.findByRole("button", { name: "Reload to see the changes" })).toBeTruthy();
  });

  it("closing the panel does not stop a running build", async () => {
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const next = stepping();
    const { fetchFn, calls } = server({
      "GET /schema": () => json(200, { tables: [], starters: [] }),
      "POST /superset/builds": () => json(201, { build: view(0) }),
      [`POST /superset/builds/${BUILD_ID}/next`]: async () => { await held; return next(); },
      "GET /superset/builds": () => json(200, { builds: [] }),
    });
    vi.stubGlobal("fetch", fetchFn);
    const { Panel } = await import("./Panel");
    const { unmount } = render(<Panel />);
    await screen.findByRole("textbox");
    await act(async () => { await driver.start("open claims", {}); });
    unmount();
    release();
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
    expect(nexts(calls)).toHaveLength(5);
  });

  it("nothing the panel holds is written to browser storage", async () => {
    const FINISHED = "1a2b3c4d-0000-4000-8000-000000000003";
    let finishing = false;
    const touched: string[] = [];
    for (const [name, storage] of [["localStorage", window.localStorage], ["sessionStorage", window.sessionStorage]] as const) {
      for (const method of ["getItem", "setItem", "removeItem", "clear", "key"] as const) {
        vi.spyOn(Object.getPrototypeOf(storage), method).mockImplementation(() => {
          touched.push(`${name}.${method}`);
          return null;
        });
      }
    }
    const opened = vi.fn(() => { touched.push("indexedDB.open"); return {} as IDBOpenDBRequest; });
    vi.stubGlobal("indexedDB", { open: opened });
    history.replaceState(null, "", "/dashboard/7/");
    host.navigate("dashboard");
    const chat = `data: ${JSON.stringify({ type: "RUN_STARTED", runId: "r" })}\n\ndata: ${JSON.stringify({ type: "RUN_FINISHED", runId: "r" })}\n\n`;
    const { fetchFn } = server({
      "GET /schema": () => json(200, { tables: [], starters: ["How many claims are open?"] }),
      "GET /api/v1/dashboard/7": () => json(200, { id: 7, result: { id: 7, dashboard_title: "Claims" } }),
      "POST /chat": () => new Response(chat, { status: 200, headers: { "content-type": "text/event-stream" } }),
      "POST /superset/builds": () => json(201, { build: view(0) }),
      [`POST /superset/builds/${BUILD_ID}/next`]: stepping(),
      "GET /superset/builds": () => json(200, { builds: finishing ? [{ build_id: FINISHED, goal: "claims by region",
        dashboard_id: 7, status: "building", created_at: "x", updated_at: "y", steps_done: 2, steps_total: 5 }] : [] }),
      [`POST /superset/builds/${FINISHED}/next`]: (() => {
        let done = 2;
        return () => json(200, { build: view(++done, { build_id: FINISHED }) });
      })(),
    });
    vi.stubGlobal("fetch", fetchFn);
    const { Panel } = await import("./Panel");
    render(<Panel />);
    fireEvent.click(await screen.findByRole("button", { name: "How many claims are open?" }));
    await vi.waitFor(() => expect(getSnapshot().turns.at(-1)?.status).toBe("complete"));
    await act(async () => { await driver.start("open claims", { dashboardId: 7 }); });
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
    // A Finish, through the banner: the list now holds a stopped build.
    finishing = true;
    cleanup();
    render(<Panel />);
    const banner = await screen.findByRole("status", { name: "Unfinished build" });
    fireEvent.click(within(banner).getByRole("button", { name: "Finish" }));
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
    expect(touched).toEqual([]);
    vi.restoreAllMocks();
  });
});

/** The panel on `path`, mnemiq and Superset answering as `handlers` say, once it shows the composer. */
async function inPanel(path: string, handlers: Record<string, Handler>) {
  history.replaceState(null, "", path);
  host.navigate(path.startsWith("/dashboard/") ? "dashboard" : "chart_list");
  const served = server({
    "GET /schema": () => json(200, { tables: [], starters: [] }),
    "GET /superset/builds": () => json(200, { builds: [] }),
    ...handlers,
  });
  vi.stubGlobal("fetch", served.fetchFn);
  const { Panel } = await import("./Panel");
  render(<Panel />);
  await screen.findByRole("textbox");
  return served;
}

/** Sends `goal` with Build, from the composer, "New dashboard" ticked when `fresh`. */
function sendBuild(goal: string, { fresh = false } = {}) {
  fireEvent.click(screen.getByText("Build", { exact: true }));
  fireEvent.change(screen.getByRole("textbox", { name: "Describe the dashboard" }), { target: { value: goal } });
  if (fresh) fireEvent.click(screen.getByRole("checkbox", { name: "New dashboard" }));
  fireEvent.click(screen.getByRole("button", { name: "Build" }));
}

const starts = (calls: Call[]) => calls.filter((c) => c.method === "POST" && c.path === "/superset/builds");
const DASHBOARDS: Record<string, Handler> = {
  "GET /api/v1/dashboard/1": () => json(200, { id: 1, result: { id: 1, dashboard_title: "One" } }),
  "GET /api/v1/dashboard/2": () => json(200, { id: 2, result: { id: 2, dashboard_title: "Two" } }),
  "GET /api/v1/dashboard/9": () => json(403, { message: "Forbidden" }),
};

describe("building from the composer", () => {
  it("Build sends what the composer says: neither off a dashboard, the dashboard on screen on one, fresh when ticked", async () => {
    const answered: Handler = () => json(200, { offer: { dashboard_id: 12, title: "Claims", reason: null, message: "covered" } });
    const off = await inPanel("/chart/list/", { "POST /superset/builds": answered });
    sendBuild("open claims");
    await vi.waitFor(() => expect(starts(off.calls)).toHaveLength(1));
    expect(starts(off.calls)[0].body).toEqual({ goal: "open claims" });
    expect(await screen.findByRole("button", { name: "Add these charts to it" })).toBeTruthy();
    cleanup();
    resetStore();
    const on = await inPanel("/dashboard/1/", { ...DASHBOARDS, "POST /superset/builds": answered });
    await screen.findByRole("region", { name: "This dashboard" });
    sendBuild("open claims");
    await vi.waitFor(() => expect(starts(on.calls)).toHaveLength(1));
    expect(starts(on.calls)[0].body).toEqual({ goal: "open claims", dashboard_id: 1 });
    cleanup();
    resetStore();
    const ticked = await inPanel("/dashboard/1/", { ...DASHBOARDS, "POST /superset/builds": answered });
    await screen.findByRole("region", { name: "This dashboard" });
    sendBuild("open claims", { fresh: true });
    await vi.waitFor(() => expect(starts(ticked.calls)).toHaveLength(1));
    expect(starts(ticked.calls)[0].body).toEqual({ goal: "open claims", fresh: true });
  });

  it("a Build goes to the dashboard on screen when it is sent, before the page context has caught up", async () => {
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const served = await inPanel("/dashboard/1/", {
      ...DASHBOARDS,
      "GET /api/v1/dashboard/2": async () => { await held; return json(200, { id: 2, result: { id: 2, dashboard_title: "Two" } }); },
      "POST /superset/builds": () => json(200, { refused: { reason: "nothing to plan", parts: [] } }),
    });
    await screen.findByRole("region", { name: "This dashboard" });
    // Superset moves to dashboard 2 with no page event, and the panel has not read the path again yet.
    history.pushState(null, "", "/dashboard/2/");
    sendBuild("open claims");
    await new Promise((done) => setTimeout(done, 30));
    expect(starts(served.calls)).toEqual([]);
    release();
    await vi.waitFor(() => expect(starts(served.calls)).toHaveLength(1));
    expect(starts(served.calls)[0].body).toEqual({ goal: "open claims", dashboard_id: 2 });
  });

  it("a Build on a dashboard the person cannot read is planned as one off a dashboard, naming nothing", async () => {
    const served = await inPanel("/dashboard/9/", { ...DASHBOARDS,
      "POST /superset/builds": () => json(200, { refused: { reason: "nothing to plan", parts: [] } }) });
    sendBuild("open claims");
    await vi.waitFor(() => expect(starts(served.calls)).toHaveLength(1));
    expect(starts(served.calls)[0].body).toEqual({ goal: "open claims" });
  });
});

describe("one build at a time", () => {
  it("a second start while one plans sends nothing and leaves the card as it was", async () => {
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const { fetchFn, calls } = server({
      "POST /superset/builds": async () => { await held; return json(201, { build: view(0) }); },
      [`POST /superset/builds/${BUILD_ID}/next`]: stepping(),
    });
    driver.attach(api(fetchFn));
    const first = driver.start("goal A", {});
    render(<BuildCard />);
    expect(screen.getByRole("status", { name: "Planning" })).toBeTruthy();
    expect(await driver.start("goal B", {})).toEqual({ failed: expect.stringMatching(/already/i) });
    expect(getSnapshot().build?.goal).toBe("goal A");
    release();
    await first;
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
    expect(starts(calls)).toHaveLength(1);
    expect(getSnapshot().build?.goal).toBe("goal A");
  });

  it("Finish is refused while another build runs, and the card is unchanged", async () => {
    const OTHER = "1a2b3c4d-0000-4000-8000-000000000002";
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const next = stepping();
    const served = await inPanel("/chart/list/", {
      "POST /superset/builds": () => json(201, { build: view(0) }),
      [`POST /superset/builds/${BUILD_ID}/next`]: async () => { await held; return next(); },
      "GET /superset/builds": () => json(200, { builds: [{ build_id: OTHER, goal: "goal X", dashboard_id: null,
        status: "building", created_at: "x", updated_at: "y", steps_done: 1, steps_total: 5 }] }),
    });
    await act(async () => { await driver.start("goal A", {}); });
    const banner = await screen.findByRole("status", { name: "Unfinished build" });
    expect((within(banner).getByRole("button", { name: "Finish" }) as HTMLButtonElement).disabled).toBe(true);
    expect(driver.resume("goal X", {}, OTHER)).toBe(false);
    expect(getSnapshot().build?.goal).toBe("goal A");
    release();
    await vi.waitFor(() => expect(getSnapshot().build?.view?.status).toBe("done"));
    expect(served.calls.filter((c) => c.path === `/superset/builds/${OTHER}/next`)).toEqual([]);
    expect(getSnapshot().build?.goal).toBe("goal A");
  });
});

describe("steps that do not move", () => {
  it("a step that made no progress waits like a busy one, then stops with mnemiq's message", async () => {
    const waits: number[] = [];
    let answers = 0;
    const stuck = view(1, { message: "Superset was busy; the step will be tried again" });
    stuck.steps[1] = { ...stuck.steps[1], status: "uncertain" };
    const { fetchFn, calls } = server({
      // One step moves, then none does: mnemiq leaves the chart's step uncertain each time.
      [`POST /superset/builds/${BUILD_ID}/next`]: () => json(200, { build: ++answers === 1 ? view(1) : stuck }),
    });
    const building = new BuildDriver({ sleep: async (ms) => { waits.push(ms); } });
    building.attach(api(fetchFn));
    building.show("open claims", {}, view(0), BUILD_ID);
    await building.run(BUILD_ID);
    expect(waits).toEqual([2000, 4000, 8000, 16000, 16000]);
    expect(nexts(calls)).toHaveLength(7);
    expect(getSnapshot().build?.stopped).toBe("Superset was busy; the step will be tried again");
    expect(building.running()).toBeNull();
  });

  it("an answer of a shape the panel does not read stops the build with a sentence", async () => {
    const { fetchFn } = server({ [`POST /superset/builds/${BUILD_ID}/next`]: () => json(200, {}) });
    const building = new BuildDriver();
    building.attach(api(fetchFn));
    building.show("open claims", {}, null, BUILD_ID);
    await building.run(BUILD_ID);
    expect(getSnapshot().build?.stopped).toMatch(/could not be read/);
    expect(building.running()).toBeNull();
  });

  it("the card shows a build answered without its check reasons, and a start answer it does not know", () => {
    const old = view(5) as Partial<BuildView>;
    delete old.check_reasons;
    act(() => { driver.show("open claims", {}, old as BuildView, BUILD_ID); });
    render(<BuildCard />);
    expect(screen.getByRole("region", { name: "Build" }).textContent).toContain("checked");
    cleanup();
    act(() => update((state) => ({ ...state, build: { goal: "x", target: {}, buildId: null, view: null, stopped: null,
      answer: { surprise: true } as never } })));
    render(<BuildCard />);
    expect(screen.getByRole("region", { name: "Build" }).textContent).toMatch(/does not read/);
  });
});
