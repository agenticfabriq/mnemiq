import { afterEach, describe, expect, it, vi } from "vitest";
import { PanelApi, TOO_OLD } from "./api";
import {
  confirmMessage,
  declineMessage,
  deleteConversation,
  listConversations,
  readConversation,
  sendMessage,
  showResults,
  startConversation,
} from "./conversation";

const TALK = "1a2b3c4d-0000-4000-8000-000000000001";
const BASE = "/extensions/agenticfabriq/mnemiq/v1/superset/conversations";

function answering(status: number, body: unknown = {}) {
  return vi.fn<typeof fetch>(async () =>
    new Response(status === 204 ? null : JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
}

const api = (fetchFn: typeof fetch) => new PanelApi("", async () => "the-csrf-token", fetchFn);

function sent(fetchFn: ReturnType<typeof answering>, call = 0) {
  const [url, init] = fetchFn.mock.calls[call];
  return { url, method: init?.method, body: init?.body ? JSON.parse(String(init.body)) : undefined };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("the conversation client", () => {
  it("lists, pages, starts, reads and deletes by the routes mnemiq serves", async () => {
    const fetchFn = answering(200, { conversations: [], next: null });
    await listConversations(api(fetchFn));
    await listConversations(api(fetchFn), "2026-11-21T10:00:00.000000+00:00|" + TALK);
    expect(sent(fetchFn, 0)).toEqual({ url: BASE, method: "GET", body: undefined });
    expect(sent(fetchFn, 1).url).toBe(`${BASE}?before=${encodeURIComponent("2026-11-21T10:00:00.000000+00:00|" + TALK)}`);

    const started = answering(201, { conversation: { id: TALK, title: "", messages: [] } });
    expect(await startConversation(api(started))).toMatchObject({ ok: true, body: { conversation: { id: TALK } } });
    expect(sent(started)).toEqual({ url: BASE, method: "POST", body: {} });

    const read = answering(200, { conversation: { id: TALK, title: "t", messages: [] } });
    await readConversation(api(read), TALK);
    expect(sent(read)).toMatchObject({ url: `${BASE}/${TALK}`, method: "GET" });

    const deleted = answering(204);
    expect(await deleteConversation(api(deleted), TALK)).toMatchObject({ ok: true, status: 204 });
    expect(sent(deleted)).toEqual({ url: `${BASE}/${TALK}/delete`, method: "POST", body: {} });
  });

  it("a message is streamed with the page it was sent from, and a choice only when made", async () => {
    const fetchFn = vi.fn<typeof fetch>(async () => new Response(
      `data: ${JSON.stringify({ type: "CUSTOM", name: "mnemiq.routed", value: { intent: "ask", text: "x", index: 0 } })}\n\n`,
      { status: 200, headers: { "content-type": "text/event-stream" } }));
    const events = [];
    for await (const event of sendMessage(api(fetchFn), TALK, { text: "how many?", page: {
      dashboardId: 7, dashboardTitle: "Claims", dashboardHasCharts: false } })) events.push(event);
    expect(events).toEqual([{ type: "CUSTOM", name: "mnemiq.routed", value: { intent: "ask", text: "x", index: 0 } }]);
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe(`${BASE}/${TALK}/messages`);
    expect(new Headers(init!.headers).get("Accept")).toBe("text/event-stream");
    expect(JSON.parse(String(init!.body))).toEqual({ text: "how many?", dashboard_id: 7, dashboard_title: "Claims",
      dashboard_has_charts: false });

    for await (const _ of sendMessage(api(fetchFn), TALK, { text: "again", page: { dashboardId: null,
      dashboardTitle: null, dashboardHasCharts: null }, choose: "build" })) void _;
    expect(JSON.parse(String(fetchFn.mock.calls[1][1]!.body))).toEqual({ text: "again", choose: "build" });
  });

  it("confirms, declines and shows results by the message's number", async () => {
    const fetchFn = answering(200, {});
    await confirmMessage(api(fetchFn), TALK, 3, { dashboard_id: 7, remove_chart_ids: [12] });
    await declineMessage(api(fetchFn), TALK, 3);
    await showResults(api(fetchFn), TALK, 1);
    expect(sent(fetchFn, 0)).toEqual({ url: `${BASE}/${TALK}/messages/3/confirm`, method: "POST",
      body: { dashboard_id: 7, remove_chart_ids: [12] } });
    expect(sent(fetchFn, 1)).toEqual({ url: `${BASE}/${TALK}/messages/3/decline`, method: "POST", body: {} });
    expect(sent(fetchFn, 2)).toEqual({ url: `${BASE}/${TALK}/messages/1/results`, method: "POST", body: {} });
  });

  it("a 404 from an older mnemiq reads as TOO_OLD", async () => {
    const answer = await listConversations(api(answering(404, { detail: "Not Found" })));
    expect(answer).toMatchObject({ ok: false, refusal: { kind: "too-old", sentence: TOO_OLD } });
    // mnemiq's own 404 for a conversation that is not the person's keeps its sentence.
    const mine = await readConversation(api(answering(404, { detail: "This person has no conversation of that id." })), TALK);
    expect(mine).toMatchObject({ ok: false, refusal: { sentence: "This person has no conversation of that id." } });
  });

  it("nothing is written to browser storage", async () => {
    const touched: string[] = [];
    for (const [name, storage] of [["localStorage", window.localStorage], ["sessionStorage", window.sessionStorage]] as const) {
      for (const method of ["getItem", "setItem", "removeItem", "clear", "key"] as const) {
        vi.spyOn(Object.getPrototypeOf(storage), method).mockImplementation(() => {
          touched.push(`${name}.${method}`);
          return null;
        });
      }
    }
    const fetchFn = answering(200, { conversations: [], next: null });
    await listConversations(api(fetchFn));
    await startConversation(api(fetchFn));
    await readConversation(api(fetchFn), TALK);
    await confirmMessage(api(fetchFn), TALK, 1, {});
    expect(touched).toEqual([]);
  });
});
