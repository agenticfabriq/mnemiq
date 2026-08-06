import { afterEach, describe, expect, it, vi } from "vitest";

import { streamChat } from "./transport";
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

  it("turns a non-200 into a RUN_ERROR rather than throwing", async () => {
    stubFetch(new Response("nope", { status: 503, statusText: "Service Unavailable" }));

    const events = await drain(streamChat("q", "thinking"));

    expect(events).toEqual([
      { type: "RUN_ERROR", message: "the engine returned 503 Service Unavailable" },
    ]);
  });

  it("skips a frame it cannot parse instead of dropping the run", async () => {
    stubFetch(sseResponse("{not json", '{"type":"RUN_FINISHED","runId":"r"}'));

    expect(await drain(streamChat("q", "thinking"))).toEqual([
      { type: "RUN_FINISHED", runId: "r" },
    ]);
  });
});
