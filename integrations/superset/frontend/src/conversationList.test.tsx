import React from "react";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { host } from "./test/core";
import { Panel } from "./Panel";
import { resetBuilds } from "./build";
import type { Message } from "./conversation";
import { getSnapshot, resetStore } from "./store";
import { resetTalk } from "./talk";
import { resetDismissed } from "./views/Unfinished";

const BASE = "/extensions/agenticfabriq/mnemiq/v1";
const TALK = "1a2b3c4d-0000-4000-8000-000000000001";
const OTHER = "1a2b3c4d-0000-4000-8000-000000000002";
const BUILD = "0f6d0c1e-6b8f-4a39-9a4e-3d5f5a9c2b10";
const PLAN = { charts: [{ key: "open_claims", title: "Open claims", family: "big_number", measures: ["open_claims"] }],
  measures: [] };

const json = (status: number, body: unknown) =>
  new Response(status === 204 ? null : JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
const stream = (...events: object[]) =>
  new Response(events.map(frame).join(""), { status: 200, headers: { "content-type": "text/event-stream" } });

type Call = { method: string; path: string; body: Record<string, unknown> | null };

/** mnemiq behind the forwarder, keeping its conversations, and Superset's dashboard API; every call recorded. Each
 * message sent is answered by `reply`, which appends what mnemiq would keep and answers its frames. */
function mnemiq(reply: (text: string, body: Record<string, unknown>, keep: (m: Partial<Message>) => number) => object[],
  dashboards: Record<number, { title: string; charts: string[] }> = {}) {
  const calls: Call[] = [];
  const kept: Record<string, Message[]> = {};
  const keep = (id: string) => (message: Partial<Message>) => {
    const list = (kept[id] ??= []);
    list.push({ index: list.length, role: "mnemiq", kind: "note", text: "", ...message } as Message);
    return list.length - 1;
  };
  const conversation = (id: string) => ({ id, title: kept[id]?.[0]?.text ?? "", created_at: "2026-11-21T10:00:00",
    updated_at: "2026-11-21T10:00:00", messages: kept[id] ?? [] });
  const handlers: [RegExp, string, (call: Call, found: RegExpExecArray) => Response | Promise<Response>][] = [
    [/^\/schema$/, "GET", () => json(200, { tables: [], starters: ["How many claims are open?"] })],
    [/^\/superset\/builds$/, "GET", () => json(200, { builds: [] })],
    [/^\/superset\/conversations$/, "GET", () => json(200, { conversations: Object.keys(kept).map((id) => ({
      id, title: conversation(id).title, updated_at: "2026-11-21T10:00:00.000000+00:00" })), next: null })],
    [/^\/superset\/conversations$/, "POST", () => { kept[TALK] ??= []; return json(201, { conversation: conversation(TALK) }); }],
    [/^\/superset\/conversations\/([^/]+)$/, "GET", (_c, m) => kept[m[1]] ? json(200, { conversation: conversation(m[1]) })
      : json(404, { detail: "This person has no conversation of that id." })],
    [/^\/superset\/conversations\/([^/]+)\/messages$/, "POST", (call, m) => {
      const body = call.body ?? {};
      const index = keep(m[1])({ role: "person", kind: "ask", text: String(body.text),
        dashboard: body.dashboard_id ? { id: Number(body.dashboard_id), title: String(body.dashboard_title) } : null });
      const events = reply(String(body.text), body, keep(m[1]));
      return stream({ type: "CUSTOM", name: "mnemiq.routed", value: { intent: "ask", text: body.text, index } },
        ...events);
    }],
    [/^\/superset\/conversations\/([^/]+)\/messages\/(\d+)\/confirm$/, "POST", (_c, m) => {
      const message = kept[m[1]][Number(m[2])];
      message.outcome = { state: "done" };
      return json(200, message.kind === "change" ? { change: { changed: [12] }, state: "done" }
        : { build: { build_id: BUILD, status: "building", dashboard_id: 7, steps: [], checks: {}, ran: null,
            message: null, plan: PLAN, check_reasons: {} } });
    }],
    [/^\/superset\/conversations\/([^/]+)\/messages\/(\d+)\/decline$/, "POST", (_c, m) => {
      kept[m[1]][Number(m[2])].outcome = { state: "declined" };
      return json(200, { state: "declined" });
    }],
    [/^\/superset\/conversations\/([^/]+)\/messages\/(\d+)\/results$/, "POST", () =>
      json(200, { columns: ["open_claims"], rows: [[9876]], row_count: 1, truncated: false, capped: false })],
    [/^\/superset\/builds\/[^/]+\/next$/, "POST", () => json(200, { build: { build_id: BUILD, status: "done",
      dashboard_id: 7, steps: [], checks: {}, ran: null, message: null, plan: PLAN, check_reasons: {} } })],
    [/^\/api\/v1\/dashboard\/(\d+)$/, "GET", (_c, m) => dashboards[Number(m[1])]
      ? json(200, { result: { id: Number(m[1]), dashboard_title: dashboards[Number(m[1])].title,
        charts: dashboards[Number(m[1])].charts } }) : json(404, {})],
  ];
  const fetchFn = vi.fn<typeof fetch>(async (input, init) => {
    const url = String(input);
    const path = (url.startsWith(BASE) ? url.slice(BASE.length) : url).split("?")[0];
    const call = { method: init?.method ?? "GET", path, body: init?.body ? JSON.parse(String(init.body)) : null };
    calls.push(call);
    for (const [pattern, method, handler] of handlers) {
      const found = pattern.exec(path);
      if (found && method === call.method) return handler(call, found);
    }
    return json(404, { detail: "Not Found" });
  });
  vi.stubGlobal("fetch", fetchFn);
  return { calls, kept };
}

const answered = (keep: (m: Partial<Message>) => number) => [
  { type: "RUN_STARTED", runId: "r" },
  { type: "TEXT_MESSAGE_CONTENT", messageId: "m", delta: "There are 412 open claims." },
  { type: "CUSTOM", name: "mnemiq.recorded", value: { index: keep({ kind: "answer", answer: {
    sentence: "There are 412 open claims.", sql: "SELECT count(*) FROM claim", grant_fingerprint: "g",
    tables_used: ["claim"] } }) } },
  { type: "RUN_FINISHED", runId: "r" },
];

function onDashboard(id: number) {
  history.replaceState(null, "", `/dashboard/${id}/`);
  host.navigate("dashboard");
}

async function opened() {
  render(<Panel />);
  return screen.findByRole("textbox", { name: "Message mnemiq" });
}

async function say(text: string) {
  const box = await screen.findByRole("textbox", { name: "Message mnemiq" });
  fireEvent.change(box, { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  await waitFor(() => expect(getSnapshot().sending).toBe(false));
}

beforeEach(() => {
  host.reset();
  resetStore();
  resetTalk();
  resetBuilds();
  resetDismissed();
  history.replaceState(null, "", "/superset/welcome/");
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("one conversation", () => {
  it("one box, no mode control", async () => {
    mnemiq(() => []);
    await opened();
    expect(screen.queryByRole("radio")).toBeNull();
    expect(screen.queryByText("Build", { selector: "label, .ant-segmented-item-label" })).toBeNull();
    expect(screen.getByRole("button", { name: "Send" })).toBeTruthy();
  });

  it("an ask is answered and kept, and the conversation survives a reload", async () => {
    const { calls } = mnemiq((_text, _body, keep) => answered(keep));
    await opened();
    await say("how many claims are open?");
    await screen.findByText("There are 412 open claims.");
    expect(calls.filter((c) => c.path === "/superset/conversations" && c.method === "POST")).toHaveLength(1);
    // A reload: the page's memory gone, the conversation read again from mnemiq.
    cleanup();
    resetStore();
    await opened();
    expect(screen.queryByText("There are 412 open claims.")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Conversations" }));
    fireEvent.click(await screen.findByRole("listitem"));
    await screen.findByText("There are 412 open claims.");
    expect(screen.getByText("how many claims are open?")).toBeTruthy();
  });

  it("a proposal writes nothing until Build it", async () => {
    const { calls } = mnemiq((text, _body, keep) => {
      const index = keep({ kind: "build", text: "I'll add 1 chart to this dashboard: Open claims.", build_id: BUILD,
        proposal: { build_id: BUILD, goal: text, dashboard_id: 7, plan: PLAN }, outcome: { state: "proposed" } });
      return [{ type: "RUN_STARTED", runId: "r" }, { type: "CUSTOM", name: "mnemiq.proposal", value: { index } },
        { type: "RUN_FINISHED", runId: "r" }];
    }, { 7: { title: "Claims", charts: [] } });
    onDashboard(7);
    await opened();
    await say("add open claims here");
    const card = await screen.findByRole("region", { name: "Proposal" });
    expect(within(within(card).getByRole("region", { name: "Plan" })).getByText(/Open claims/)).toBeTruthy();
    expect(calls.some((c) => c.path.endsWith("/confirm") || c.path.endsWith("/next"))).toBe(false);
    fireEvent.click(within(card).getByRole("button", { name: "Build it" }));
    await waitFor(() => expect(calls.some((c) => c.path === `/superset/builds/${BUILD}/next`)).toBe(true));
    const confirm = calls.find((c) => c.path.endsWith("/confirm"))!;
    expect(confirm.body).toEqual({ dashboard_id: 7 });
  });

  it("Apply is offered only on the previewed dashboard", async () => {
    mnemiq((_text, _body, keep) => {
      const index = keep({ kind: "change", text: "To change: Sales: at most 5 rows.", proposal: {
        dashboard_id: 7, dashboard_changed_on: "t", edits: [{ chart_id: 12, title: "Sales", changed_on: "t",
          said: "Sales: at most 5 rows", digest: "d" }], would_remove: [], add_chart: false, left: [], dropped: [],
        not_editor: [], message: "To change", reason: null }, outcome: { state: "proposed" } });
      return [{ type: "RUN_STARTED", runId: "r" }, { type: "CUSTOM", name: "mnemiq.preview", value: { index } },
        { type: "RUN_FINISHED", runId: "r" }];
    }, { 7: { title: "Sales", charts: ["Sales"] }, 8: { title: "Other", charts: ["x"] } });
    onDashboard(7);
    await opened();
    await say("top five only");
    const card = await screen.findByRole("region", { name: "Preview" });
    expect(within(card).getByText("Sales: at most 5 rows")).toBeTruthy();
    await waitFor(() => expect((within(card).getByRole("button", { name: "Apply" }) as HTMLButtonElement).disabled).toBe(false));
    act(() => onDashboard(8));
    await waitFor(() => expect((within(screen.getByRole("region", { name: "Preview" }))
      .getByRole("button", { name: "Apply" }) as HTMLButtonElement).disabled).toBe(true));
    expect(screen.getByRole("link", { name: "Open the dashboard this is for" })).toBeTruthy();
  });

  it("Show results renders the rows and keeps none after closing", async () => {
    mnemiq((_text, _body, keep) => answered(keep));
    await opened();
    await say("how many claims are open?");
    fireEvent.click(await screen.findByRole("button", { name: "Show results" }));
    await screen.findByText("9876");
    // The rows are the card's alone: nothing the panel's store holds names them.
    expect(JSON.stringify(getSnapshot())).not.toContain("9876");
    cleanup();
    await opened();
    expect(screen.queryByText("9876")).toBeNull();
    expect(await screen.findByRole("button", { name: "Show results" })).toBeTruthy();
  });

  it("a clarify resends with the chosen intent", async () => {
    const { calls } = mnemiq((_text, body, keep) => {
      if (body.choose) return answered(keep);
      const index = keep({ kind: "clarify", text: "Which do you mean?", clarify: { question: "Which do you mean?",
        choices: ["ask", "build", "change"] } });
      return [{ type: "RUN_STARTED", runId: "r" }, { type: "CUSTOM", name: "mnemiq.clarify", value: { index } },
        { type: "RUN_FINISHED", runId: "r" }];
    });
    await opened();
    await say("claims");
    fireEvent.click(await screen.findByRole("button", { name: "Answer it" }));
    await waitFor(() => expect(calls.filter((c) => c.path.endsWith("/messages"))).toHaveLength(2));
    expect(calls.filter((c) => c.path.endsWith("/messages"))[1].body).toEqual({ text: "claims", choose: "ask" });
  });

  it("an empty dashboard opens with the describe placeholder", async () => {
    mnemiq(() => [], { 7: { title: "Untitled", charts: [] }, 8: { title: "Full", charts: ["a"] } });
    onDashboard(7);
    const box = await opened();
    await waitFor(() => expect(box.getAttribute("placeholder")).toBe("Describe what this dashboard should show"));
    act(() => onDashboard(8));
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message mnemiq" }).getAttribute("placeholder"))
      .not.toBe("Describe what this dashboard should show"));
  });

  it("the list pages and New starts a conversation", async () => {
    const { calls, kept } = mnemiq((_text, _body, keep) => answered(keep));
    kept[OTHER] = [{ index: 0, role: "person", kind: "ask", text: "an older question" }];
    await opened();
    fireEvent.click(screen.getByRole("button", { name: "Conversations" }));
    fireEvent.click(await screen.findByRole("listitem"));
    await screen.findByText("an older question");
    fireEvent.click(screen.getByRole("button", { name: "New" }));
    expect(screen.queryByText("an older question")).toBeNull();
    await say("how many claims are open?");
    expect(calls.filter((c) => c.path === "/superset/conversations" && c.method === "POST")).toHaveLength(1);
    expect(getSnapshot().conversationId).toBe(TALK);
  });

  it("Not now sets the proposal aside", async () => {
    const { calls } = mnemiq((text, _body, keep) => {
      const index = keep({ kind: "build", text: "I'll add 1 chart to a new dashboard: Open claims.", build_id: BUILD,
        proposal: { build_id: BUILD, goal: text, dashboard_id: null, plan: PLAN }, outcome: { state: "proposed" } });
      return [{ type: "RUN_STARTED", runId: "r" }, { type: "CUSTOM", name: "mnemiq.proposal", value: { index } },
        { type: "RUN_FINISHED", runId: "r" }];
    });
    await opened();
    await say("build a claims dashboard");
    fireEvent.click(within(await screen.findByRole("region", { name: "Proposal" })).getByRole("button", { name: "Not now" }));
    await screen.findByText("Not now.");
    expect(calls.some((c) => c.path.endsWith("/decline"))).toBe(true);
    expect(calls.some((c) => c.path.endsWith("/confirm"))).toBe(false);
  });

  it("a message goes from the dashboard on screen when it is sent, before the page context has caught up", async () => {
    const { calls } = mnemiq((_text, _body, keep) => answered(keep), { 1: { title: "One", charts: ["a"] },
      2: { title: "Two", charts: [] } });
    onDashboard(1);
    await opened();
    // Superset moves to dashboard 2 with no page event: the path changes, the page context has not caught up yet.
    history.replaceState(null, "", "/dashboard/2/");
    await say("add a chart here");
    const sent = calls.filter((c) => c.path.endsWith("/messages"))[0].body;
    expect(sent).toEqual({ text: "add a chart here", dashboard_id: 2, dashboard_title: "Two", dashboard_has_charts: false });
  });

  it("a message on a dashboard the person cannot read is sent from no dashboard, naming nothing", async () => {
    const { calls } = mnemiq((_text, _body, keep) => answered(keep));
    onDashboard(9);
    await opened();
    await say("how many claims are open?");
    expect(calls.filter((c) => c.path.endsWith("/messages"))[0].body).toEqual({ text: "how many claims are open?" });
  });

  it("a starter is sent as the conversation's first message", async () => {
    const { calls } = mnemiq((_text, _body, keep) => answered(keep));
    await opened();
    fireEvent.click(await screen.findByRole("button", { name: "How many claims are open?" }));
    await screen.findByText("There are 412 open claims.");
    expect(calls.filter((c) => c.path.endsWith("/messages"))[0].body).toEqual({ text: "How many claims are open?" });
    expect(screen.queryByRole("region", { name: "Starters" })).toBeNull();
  });
});
