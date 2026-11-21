import { describe, expect, it, vi } from "vitest";
import { PanelApi, refusalOf, TOO_OLD } from "./api";

const NOT_A_BUILDER = "Superset refused this user's queries. Building with mnemiq needs Superset's Alpha and sql_lab roles.";

function answering(status: number, body: unknown = {}, headers: Record<string, string> = {}) {
  return vi.fn<typeof fetch>(async () =>
    new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } }));
}

describe("the panel's API client", () => {
  it("a POST carries the CSRF token and no Authorization header", async () => {
    const fetchFn = answering(201, { build: { build_id: "b" } });
    const api = new PanelApi("/analytics", async () => "the-csrf-token", fetchFn);
    const answer = await api.post("/superset/builds", { goal: "a goal" });
    expect(answer).toEqual({ ok: true, status: 201, body: { build: { build_id: "b" } } });
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("/analytics/extensions/agenticfabriq/mnemiq/v1/superset/builds");
    const headers = new Headers(init!.headers);
    expect(headers.get("X-CSRFToken")).toBe("the-csrf-token");
    expect(headers.has("Authorization")).toBe(false);
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(init).toMatchObject({ method: "POST", credentials: "same-origin", redirect: "manual",
      body: JSON.stringify({ goal: "a goal" }) });
  });

  it("a GET carries no CSRF token, and is same-origin with no redirect followed", async () => {
    const fetchFn = answering(200, { tables: [], starters: [] });
    await new PanelApi("", async () => "t", fetchFn).get("/schema");
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("/extensions/agenticfabriq/mnemiq/v1/schema");
    expect(new Headers(init!.headers).has("X-CSRFToken")).toBe(false);
    expect(init).toMatchObject({ method: "GET", credentials: "same-origin", redirect: "manual" });
  });

  it("a request that never reaches Superset is unavailable", async () => {
    const api = new PanelApi("", async () => "t", vi.fn<typeof fetch>(async () => { throw new TypeError("Failed to fetch"); }));
    const answer = await api.get("/schema");
    expect(answer).toMatchObject({ ok: false, refusal: { kind: "unavailable" } });
  });

  it("an answer no kind names is still a refusal, with mnemiq's sentence", async () => {
    const api = new PanelApi("", async () => "t", answering(422, { detail: "the goal is empty" }));
    expect(await api.post("/superset/builds", { goal: "" })).toEqual({ ok: false, status: 422,
      refusal: { kind: "refused", sentence: "the goal is empty" } });
  });
});

describe("refusalOf", () => {
  const response = (status: number, headers: Record<string, string> = {}) => new Response(null, { status, headers });
  const cases: [string, Response, string | null, { kind: string; sentence?: string } | null][] = [
    ["Superset's own 401", response(401), null, { kind: "signed-out" }],
    ["a sign-in redirect", Object.defineProperty(response(200), "type", { value: "opaqueredirect" }), null,
      { kind: "signed-out" }],
    ["mnemiq's 401, refusing the extension's token", response(401, { "www-authenticate": "Bearer" }),
      "the token's signature did not verify", { kind: "unavailable", sentence: "the token's signature did not verify" }],
    ["the roles sentence", response(403), NOT_A_BUILDER, { kind: "not-a-builder", sentence: NOT_A_BUILDER }],
    ["an ask token on a build route", response(403, { "www-authenticate": 'Bearer error="insufficient_scope"' }),
      "this route needs a build token", { kind: "not-for-building", sentence: "this route needs a build token" }],
    ["any other 403", response(403), "you may not", { kind: "not-permitted", sentence: "you may not" }],
    ["a busy step", response(409), "a step is running", { kind: "busy", sentence: "a step is running" }],
    ["502", response(502), "mnemiq did not answer", { kind: "unavailable", sentence: "mnemiq did not answer" }],
    ["503", response(503), null, { kind: "unavailable" }],
    ["504", response(504), "mnemiq took too long", { kind: "unavailable", sentence: "mnemiq took too long" }],
    ["a mnemiq without the route", response(404), "Not Found", { kind: "too-old", sentence: TOO_OLD }],
    ["another 404", response(404), "no such build", null],
    ["a 200", response(200), null, null],
  ];
  it.each(cases)("%s", (_name, given, detail, expected) => {
    const found = refusalOf(given, detail);
    if (expected === null) return expect(found).toBeNull();
    expect(found?.kind).toBe(expected.kind);
    expect(found?.sentence).toBe(expected.sentence ?? found?.sentence);
    expect(found?.sentence.length).toBeGreaterThan(0);
  });
});
