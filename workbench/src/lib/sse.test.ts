import { describe, expect, it } from "vitest";

import { SseDecoder } from "./sse";

const frame = (payload: object) => `data: ${JSON.stringify(payload)}\n\n`;

describe("SseDecoder", () => {
  it("emits nothing until a frame is terminated", () => {
    const decoder = new SseDecoder();
    expect(decoder.push('data: {"type":"RUN_STARTED"}')).toEqual([]);
    expect(decoder.push("\n\n")).toEqual(['{"type":"RUN_STARTED"}']);
  });

  it("reassembles a frame split mid-payload across chunks", () => {
    const decoder = new SseDecoder();
    const whole = frame({ type: "TEXT_MESSAGE_CONTENT", delta: "There are 2 claims." });
    const cut = Math.floor(whole.length / 2);

    expect(decoder.push(whole.slice(0, cut))).toEqual([]);
    const out = decoder.push(whole.slice(cut));

    expect(out).toHaveLength(1);
    expect(JSON.parse(out[0]!).delta).toBe("There are 2 claims.");
  });

  it("survives a split that lands between the two terminating newlines", () => {
    const decoder = new SseDecoder();
    expect(decoder.push('data: {"type":"RUN_FINISHED"}\n')).toEqual([]);
    expect(decoder.push("\n")).toEqual(['{"type":"RUN_FINISHED"}']);
  });

  it("emits every frame when several arrive in one chunk", () => {
    const decoder = new SseDecoder();
    const out = decoder.push(
      frame({ type: "RUN_STARTED" }) +
        frame({ type: "TEXT_MESSAGE_START" }) +
        frame({ type: "RUN_FINISHED" }),
    );
    expect(out.map((d) => JSON.parse(d).type)).toEqual([
      "RUN_STARTED",
      "TEXT_MESSAGE_START",
      "RUN_FINISHED",
    ]);
  });

  it("ignores keep-alive comments but keeps the frames around them", () => {
    const decoder = new SseDecoder();
    const out = decoder.push(
      ": keep-alive\n\n: keep-alive\n\n" + frame({ type: "RUN_FINISHED" }),
    );
    expect(out).toEqual(['{"type":"RUN_FINISHED"}']);
  });

  it("keeps an unterminated tail pending rather than emitting it", () => {
    const decoder = new SseDecoder();
    decoder.push(frame({ type: "RUN_STARTED" }) + 'data: {"type":"RUN_F');
    expect(decoder.pending).toBe('data: {"type":"RUN_F');
  });

  it("joins a multi-line data field and tolerates CRLF", () => {
    const decoder = new SseDecoder();
    expect(decoder.push("data: one\r\ndata: two\r\n\r\n")).toEqual(["one\ntwo"]);
  });
});
