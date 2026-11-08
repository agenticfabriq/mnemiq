import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchScope, fetchWhoami, NOTICE, Refused, streamChat, type Refusal } from "./transport";
import type { HistoryTurn } from "./types";

function sseResponse(...frames: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encode = new TextEncoder();
      for (const frame of frames) controller.enqueue(encode.encode(`data: ${frame}\n\n`));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

function stubFetch(response: Response) {
  const fetchMock = vi.fn().mockResolvedValue(response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function drain(stream: AsyncGenerator<unknown>) {
  const out = [];
  for await (const event of stream) out.push(event);
  return out;
}

const TURN: HistoryTurn = {
  question: "which region has the most claims?",
  sql: "SELECT region FROM policy",
  tables_used: ["policy"],
  columns: ["region"],
  rows: [["west"]],
  grant_fingerprint: "fp-analyst",
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("streamChat", () => {
  it("sends the history the caller passed", async () => {
    const fetchMock = stubFetch(sseResponse('{"type":"RUN_FINISHED","runId":"r"}'));

    await drain(streamChat("and how many claims?", "thinking", [TURN]));

    const body = JSON.parse(fetchMock.mock.calls[0]![1].body);
    expect(body).toEqual({
      question: "and how many claims?",
      mode: "thinking",
      history: [TURN],
    });
  });

  it("sends an empty history when the caller passes none", async () => {
    const fetchMock = stubFetch(sseResponse('{"type":"RUN_FINISHED","runId":"r"}'));

    await drain(streamChat("how many?", "instant"));

    expect(JSON.parse(fetchMock.mock.calls[0]![1].body).history).toEqual([]);
  });

  it("turns a non-200 into a run of its own that failed, rather than throwing", async () => {
    // The run never started on the engine, so it has no assistant turn for the error to land
    // on. Without one, RUN_ERROR patched the previous question's answer, or nothing at all.
    stubFetch(new Response("nope", { status: 500, statusText: "Internal Server Error" }));

    const events = await drain(streamChat("q", "thinking"));

    expect(events).toEqual([
      { type: "RUN_STARTED", runId: "" },
      { type: "RUN_ERROR", message: "the engine returned 500 Internal Server Error" },
    ]);
  });

  it("skips a frame it cannot parse instead of dropping the run", async () => {
    stubFetch(sseResponse("{not json", '{"type":"RUN_FINISHED","runId":"r"}'));

    expect(await drain(streamChat("q", "thinking"))).toEqual([
      { type: "RUN_FINISHED", runId: "r" },
    ]);
  });
});

/** What a redirect looks like to a fetch made with `redirect: "manual"`: status 0, no body. */
const OPAQUE_REDIRECT = {
  type: "opaqueredirect",
  status: 0,
  ok: false,
  statusText: "",
  headers: new Headers(),
  body: null,
} as unknown as Response;

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const refusalOf = async (promise: Promise<unknown>): Promise<Refusal | "none"> => {
  try {
    await promise;
    return "none";
  } catch (error) {
    if (error instanceof Refused) return error.refusal;
    throw error;
  }
};

/** Each answer that says something about the session rather than the question. */
const SESSION_ANSWERS: [string, () => Response, Refusal][] = [
  ["a 401", () => new Response("", { status: 401 }), "signed-out"],
  ["a redirect to the identity provider", () => OPAQUE_REDIRECT, "signed-out"],
  ["a 403", () => new Response("", { status: 403 }), "not-permitted"],
  [
    "a 403 for a missing scope",
    () =>
      new Response("", {
        status: 403,
        headers: { "www-authenticate": 'Bearer error="insufficient_scope", scope="db.read"' },
      }),
    "insufficient-scope",
  ],
  ["a 503", () => new Response("", { status: 503, statusText: "Service Unavailable" }), "unavailable"],
];

describe("fetchWhoami", () => {
  it("reads an engine without the route as one identity", async () => {
    stubFetch(new Response("Not Found", { status: 404 }));

    expect(await fetchWhoami()).toBeNull();
  });

  it("reads the identity the engine reports", async () => {
    stubFetch(json({ principal_id: "alice@example.com", tenant_id: "tenant-a", verified: true }));

    expect(await fetchWhoami()).toEqual({
      principal_id: "alice@example.com",
      tenant_id: "tenant-a",
      verified: true,
    });
  });

  it("reads the settings identity as unverified", async () => {
    stubFetch(json({ principal_id: "analyst", tenant_id: "local", verified: false }));

    expect((await fetchWhoami())!.verified).toBe(false);
  });

  it("treats a missing verified flag as verified", async () => {
    // Fail safe: read as unverified, a verified user would inherit the legacy history.
    stubFetch(json({ principal_id: "alice@example.com", tenant_id: "tenant-a" }));

    expect((await fetchWhoami())!.verified).toBe(true);
  });

  it("tolerates a null tenant", async () => {
    stubFetch(json({ principal_id: "alice@example.com", tenant_id: null, verified: true }));

    expect((await fetchWhoami())!.tenant_id).toBeNull();
  });

  it("reads a tenant that is neither a string nor null as the service failing", async () => {
    // Read as null, the same principal in two tenants would share one namespace, and a
    // malformed answer could set off the purge of another identity's threads.
    for (const tenant_id of [undefined, 7, { id: "tenant-a" }]) {
      stubFetch(json({ principal_id: "alice@example.com", tenant_id, verified: true }));
      expect(await refusalOf(fetchWhoami())).toBe("unavailable");
    }
  });

  it("reads an answer that names nobody as the service failing", async () => {
    stubFetch(json({ tenant_id: "tenant-a", verified: true }));
    expect(await refusalOf(fetchWhoami())).toBe("unavailable");

    stubFetch(new Response("<!doctype html>", { status: 200 }));
    expect(await refusalOf(fetchWhoami())).toBe("unavailable");

    stubFetch(new Response("", { status: 500 }));
    expect(await refusalOf(fetchWhoami())).toBe("unavailable");
  });

  for (const [what, response, refusal] of SESSION_ANSWERS) {
    it(`reads ${what} as ${refusal}`, async () => {
      stubFetch(response());
      expect(await refusalOf(fetchWhoami())).toBe(refusal);
    });
  }

  it("reads a network failure as the service unavailable, never as signed out", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    expect(await refusalOf(fetchWhoami())).toBe("unavailable");
  });
});

describe("fetchScope", () => {
  for (const [what, response, refusal] of SESSION_ANSWERS) {
    it(`reads ${what} as ${refusal}`, async () => {
      stubFetch(response());
      expect(await refusalOf(fetchScope())).toBe(refusal);
    });
  }

  it("reads a network failure as the service unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    expect(await refusalOf(fetchScope())).toBe("unavailable");
  });

  it("still reads any other failure as an empty scope", async () => {
    stubFetch(new Response("", { status: 500 }));

    expect(await fetchScope()).toEqual({ tables: [], starters: [] });
  });
});

describe("streamChat before the stream starts", () => {
  it("ends the run by throwing when the session is signed out", async () => {
    // Thrown, not yielded: the page is about to purge this user's threads, and no event may
    // reach a thread after that.
    for (const response of [new Response("", { status: 401 }), OPAQUE_REDIRECT]) {
      stubFetch(response);
      expect(await refusalOf(drain(streamChat("q", "thinking")))).toBe("signed-out");
    }
  });

  for (const [what, response, refusal] of SESSION_ANSWERS.filter(([, , r]) => r !== "signed-out")) {
    it(`answers ${what} with a failed run that says so`, async () => {
      stubFetch(response());

      expect(await drain(streamChat("q", "thinking"))).toEqual([
        { type: "RUN_STARTED", runId: "" },
        {
          type: "RUN_ERROR",
          message: NOTICE[refusal as Exclude<Refusal, "signed-out">],
          // Only a service problem is coded: the card says so instead of "still serving".
          ...(refusal === "unavailable" && { code: "unavailable" }),
        },
      ]);
    });
  }

  it("answers a network failure with a failed run saying the service is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    expect(await drain(streamChat("q", "thinking"))).toEqual([
      { type: "RUN_STARTED", runId: "" },
      { type: "RUN_ERROR", message: NOTICE.unavailable, code: "unavailable" },
    ]);
  });

  it("lets a cancelled request end the run as it always did", async () => {
    const abort = new DOMException("aborted", "AbortError");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(abort));

    await expect(drain(streamChat("q", "thinking"))).rejects.toBe(abort);
  });
});

describe("every /v1 request", () => {
  it("is made without following a redirect, so a sign-in redirect is seen as one", async () => {
    // Followed, a redirect to the identity provider is a cross-origin fetch the browser
    // refuses, and the refusal reads as a network failure: an outage, not a sign-out.
    const fetchMock = stubFetch(new Response("", { status: 404 }));
    await fetchWhoami();
    fetchMock.mockResolvedValue(json({ tables: [], starters: [] }));
    await fetchScope();
    fetchMock.mockResolvedValue(sseResponse('{"type":"RUN_FINISHED","runId":"r"}'));
    await drain(streamChat("q", "thinking"));

    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.redirect])).toEqual([
      ["/v1/whoami", "manual"],
      ["/v1/schema", "manual"],
      ["/v1/chat", "manual"],
    ]);
  });
});
