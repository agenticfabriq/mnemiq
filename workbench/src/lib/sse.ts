/**
 * Incremental SSE decoding.
 *
 * A chunk boundary can fall anywhere -- mid-frame, mid-line, even between the two
 * newlines that terminate a frame -- so nothing is emitted until a full frame is in
 * the buffer. The engine sends `: keep-alive` comments while it works; those carry
 * no data and must not surface as events.
 */

const FRAME_END = "\n\n";

export class SseDecoder {
  private buffer = "";

  /** Feed a chunk; get back the `data:` payload of every frame it completed. */
  push(chunk: string): string[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const out: string[] = [];
    let end: number;
    while ((end = this.buffer.indexOf(FRAME_END)) !== -1) {
      const frame = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + FRAME_END.length);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice("data:".length).replace(/^ /, ""))
        .join("\n");
      if (data) out.push(data);
    }
    return out;
  }

  /** Whatever has arrived but is not yet a complete frame. */
  get pending(): string {
    return this.buffer;
  }
}

/** Decode a fetch body into SSE data payloads. */
export async function* readSse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new SseDecoder();
  const utf8 = new TextDecoder();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const data of decoder.push(utf8.decode(value, { stream: true }))) {
        yield data;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
