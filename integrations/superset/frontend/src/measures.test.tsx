import React from "react";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PanelApi } from "./api";
import { measureValueText, type MeasureItem } from "./measures";
import { Measures } from "./views/Measures";
import { getSnapshot, resetStore } from "./store";

const BASE = "/extensions/agenticfabriq/mnemiq/v1";
const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

function measure(fields: Partial<MeasureItem> = {}): MeasureItem {
  return { name: "open_claims", label: "Open claims", expression: "COUNT(*)", kind: "count", currency: null,
    description: "", asked_by: "", dataset_id: 3, saved_metric: null, value: 412, value_note: null,
    charts: [{ chart_id: 101, title: "Open claims", check: "checked" }, { chart_id: 102, title: "Claims by region", check: null }],
    ...fields };
}

type Call = { method: string; path: string; body: unknown };

function server(handlers: Record<string, (call: Call) => Response | Promise<Response>>) {
  const calls: Call[] = [];
  const fetchFn = vi.fn<typeof fetch>(async (input, init) => {
    const path = String(input).slice(BASE.length);
    const call = { method: init?.method ?? "GET", path, body: init?.body ? JSON.parse(String(init.body)) : null };
    calls.push(call);
    const handler = handlers[`${call.method} ${path}`];
    return handler ? handler(call) : json(404, { detail: "Not Found" });
  });
  return { api: new PanelApi("", async () => "csrf", fetchFn), calls };
}

const LIST = "/superset/dashboards/7/measures";
const open = () => fireEvent.click(screen.getByRole("button", { name: /Measures/ }));

beforeEach(() => resetStore());
afterEach(cleanup);

describe("measureValueText", () => {
  it("each kind reads as its kind", () => {
    expect(measureValueText(1234.5, "currency", "USD", "en-US")).toBe("$1,234.50");
    expect(measureValueText(0.1234, "percent", null, "en-US")).toBe("12.3%");
    expect(measureValueText(1234567, "count", null, "en-US")).toBe("1,234,567");
    expect(measureValueText(1234.567, "number", null, "en-US")).toBe("1,234.57");
    expect(measureValueText(12.5, "currency", null, "en-US")).toBe("12.50");
  });

  it("a measure with no value says so, never 0", () => {
    for (const kind of ["currency", "percent", "count", "number"] as const) {
      expect(measureValueText(null, kind, "USD", "en-US")).toBe("no value");
    }
  });
});

describe("the dashboard's measures", () => {
  it("values are read each time the section opens, never reused", async () => {
    let value = 412;
    const { api, calls } = server({ [`GET ${LIST}`]: () => json(200, { measures: [measure({ value: value++ })] }) });
    render(<Measures api={api} dashboardId={7} />);
    expect(calls).toEqual([]);
    open();
    await screen.findByText("412");
    open();
    expect(screen.queryByText("412")).toBeNull();
    open();
    await screen.findByText("413");
    expect(calls.filter((c) => c.path === LIST)).toHaveLength(2);
  });

  it("a measure with no value says so in the list", async () => {
    const { api } = server({ [`GET ${LIST}`]: () => json(200, { measures: [measure({ value: null, value_note: "Superset refused this query" })] }) });
    render(<Measures api={api} dashboardId={7} />);
    open();
    const row = await screen.findByRole("listitem");
    expect(row.textContent).toContain("no value");
    expect(row.textContent).toContain("Superset refused this query");
    expect(row.textContent).not.toMatch(/\b0\b/);
  });

  it("Remove names the charts it deletes before it deletes them, and sends those ids", async () => {
    const { api, calls } = server({
      [`GET ${LIST}`]: () => json(200, { measures: [measure()] }),
      [`POST ${LIST}/open_claims/remove`]: () => json(200, { changed: [101], left_by_hand: [], not_editor: [
        { chart_id: 102, dataset_id: 3, editors: ["Bo Owner"] }], blocked_by_report: [], could_not_read: [],
        message: "Deleted 1 chart, moved to Superset's trash; 1 chart you do not edit was taken off this dashboard." }),
    });
    render(<Measures api={api} dashboardId={7} />);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Remove" }));
    const dialog = await screen.findByRole("dialog");
    expect(dialog.textContent).toContain("Open claims");
    expect(dialog.textContent).toContain("Claims by region");
    // What Remove does, said before it is done: the charts are deleted, not the measure taken off them.
    expect(dialog.textContent).toMatch(/deleted/);
    expect(dialog.textContent).toMatch(/taken off this dashboard instead/);
    expect(calls.filter((c) => c.method === "POST")).toEqual([]);
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete these charts" }));
    await screen.findByText("Deleted 1 chart, moved to Superset's trash; 1 chart you do not edit was taken off this dashboard.");
    const result = screen.getByRole("status", { name: "Result" });
    expect(result.textContent).toMatch(/Deleted: Open claims/);
    expect(result.textContent).toMatch(/Taken off this dashboard \(you do not edit it\): Claims by region; its editors: Bo Owner/);
    expect(result.textContent).not.toMatch(/Changed:/);
    expect(calls.find((c) => c.method === "POST")).toEqual({ method: "POST", path: `${LIST}/open_claims/remove`,
      body: { chart_ids: [101, 102] } });
  });

  it("a remove refused because the charts changed shows the new list", async () => {
    const { api } = server({
      [`GET ${LIST}`]: () => json(200, { measures: [measure()] }),
      [`POST ${LIST}/open_claims/remove`]: () => json(409, { detail: { message: "The measure's charts changed; look again.",
        charts: [{ chart_id: 101, title: "Open claims", check: "checked" }, { chart_id: 103, title: "Claims this week", check: null }] } }),
    });
    render(<Measures api={api} dashboardId={7} />);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Remove" }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Delete these charts" }));
    await screen.findByText("The measure's charts changed; look again.");
    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain("Claims this week");
    expect(dialog.textContent).not.toContain("Claims by region");
  });

  it("Edit shows which copies changed and which were left as changed by hand", async () => {
    const { api, calls } = server({
      [`GET ${LIST}`]: () => json(200, { measures: [measure()] }),
      [`POST ${LIST}/open_claims/edit`]: () => json(200, { changed: [101], left_by_hand: [102], not_editor: [],
        blocked_by_report: [], could_not_read: [], message: "Open claims was changed on 1 chart." }),
    });
    render(<Measures api={api} dashboardId={7} />);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("Label"), { target: { value: "Claims still open" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save changes" }));
    await screen.findByText("Open claims was changed on 1 chart.");
    expect(calls.find((c) => c.method === "POST")?.body).toEqual({ label: "Claims still open", expression: "COUNT(*)",
      kind: "count", currency: null, description: "" });
    const result = screen.getByRole("status", { name: "Result" });
    expect(result.textContent).toMatch(/Changed: Open claims/);
    expect(result.textContent).toMatch(/Left as changed by hand: Claims by region/);
  });

  it("Save where the builder does not edit the dataset names its editors and changes nothing", async () => {
    const { api } = server({
      [`GET ${LIST}`]: () => json(200, { measures: [measure()] }),
      [`POST ${LIST}/open_claims/save`]: () => json(200, { changed: [], left_by_hand: [], not_editor: [
        { dataset_id: 3, editors: ["Ana Admin", "Bo Owner"] }], blocked_by_report: [], could_not_read: [],
        message: "You are not an editor of this measure's dataset, so it was not saved." }),
    });
    render(<Measures api={api} dashboardId={7} />);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Save to dataset" }));
    await screen.findByText("You are not an editor of this measure's dataset, so it was not saved.");
    const result = screen.getByRole("status", { name: "Result" });
    expect(result.textContent).toContain("Ana Admin, Bo Owner");
    expect(result.textContent).not.toMatch(/Changed:/);
  });

  it("no measure value reaches the module store", async () => {
    const { api } = server({ [`GET ${LIST}`]: () => json(200, { measures: [measure({ value: 987654.321 })] }) });
    render(<Measures api={api} dashboardId={7} />);
    open();
    await screen.findByText("987,654");
    expect(JSON.stringify(getSnapshot())).not.toMatch(/98765/);
  });

  it("an action in flight sends once, and holds the other actions until it answers", async () => {
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const { api, calls } = server({
      [`GET ${LIST}`]: () => json(200, { measures: [measure()] }),
      [`POST ${LIST}/open_claims/remove`]: async () => { await held; return json(200, { changed: [101, 102],
        left_by_hand: [], not_editor: [], blocked_by_report: [], could_not_read: [], message: "Deleted 2 charts." }); },
      [`POST ${LIST}/open_claims/edit`]: async () => { await held; return json(200, { changed: [101], left_by_hand: [],
        not_editor: [], blocked_by_report: [], could_not_read: [], message: "Changed." }); },
    });
    render(<Measures api={api} dashboardId={7} />);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Remove" }));
    const confirm = within(await screen.findByRole("dialog")).getByRole("button", { name: "Delete these charts" });
    fireEvent.click(confirm);
    fireEvent.click(confirm);
    await new Promise((done) => setTimeout(done, 20));
    expect(calls.filter((c) => c.path.endsWith("/remove"))).toHaveLength(1);
    for (const name of ["Edit", "Remove", "Save to dataset"]) {
      expect((screen.getAllByRole("button", { name })[0] as HTMLButtonElement).disabled, name).toBe(true);
    }
    release();
    await screen.findByText("Deleted 2 charts.");
    expect(calls.filter((c) => c.path.endsWith("/remove"))).toHaveLength(1);
  });

  it("Edit stays open until mnemiq answers, so a second edit starts from the saved definition", async () => {
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const { api, calls } = server({
      [`GET ${LIST}`]: () => json(200, { measures: [measure()] }),
      [`POST ${LIST}/open_claims/edit`]: async () => { await held; return json(200, { changed: [101], left_by_hand: [],
        not_editor: [], blocked_by_report: [], could_not_read: [], message: "Changed." }); },
    });
    render(<Measures api={api} dashboardId={7} />);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Save changes" }));
    fireEvent.click(within(dialog).getByRole("button", { name: "Save changes" }));
    await new Promise((done) => setTimeout(done, 20));
    expect(calls.filter((c) => c.path.endsWith("/edit"))).toHaveLength(1);
    expect(screen.getByRole("dialog")).toBeTruthy();
    release();
    await screen.findByText("Changed.");
  });

  it("each measure shows its definition and each chart's check; a saved metric offers no Edit", async () => {
    const { api } = server({ [`GET ${LIST}`]: () => json(200, { measures: [
      measure(), measure({ name: "premium@3", label: "Premium", expression: "SUM(premium)", saved_metric: "premium",
        charts: [{ chart_id: 103, title: "Premium by region", check: "mismatch" }] })] }) });
    render(<Measures api={api} dashboardId={7} />);
    open();
    const [counted, saved] = await screen.findAllByRole("listitem");
    expect(counted.textContent).toContain("COUNT(*)");
    expect(counted.textContent).toMatch(/Open claims: checked/);
    expect(saved.textContent).toMatch(/Premium by region: numbers differ from mnemiq's/);
    expect(within(counted).getByRole("button", { name: "Edit" })).toBeTruthy();
    expect(within(saved).queryByRole("button", { name: "Edit" })).toBeNull();
    expect(within(saved).queryByRole("button", { name: "Save to dataset" })).toBeNull();
  });

  it("an action answered after the section closed reads nothing more", async () => {
    let release = () => {};
    const held = new Promise<void>((done) => { release = done; });
    const { api, calls } = server({
      [`GET ${LIST}`]: () => json(200, { measures: [measure()] }),
      [`POST ${LIST}/open_claims/save`]: async () => { await held; return json(200, { changed: [], left_by_hand: [],
        not_editor: [], blocked_by_report: [], could_not_read: [], message: "Saved." }); },
    });
    render(<Measures api={api} dashboardId={7} />);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Save to dataset" }));
    open();
    release();
    await new Promise((done) => setTimeout(done, 30));
    expect(calls.filter((c) => c.path === LIST)).toHaveLength(1);
  });
});
