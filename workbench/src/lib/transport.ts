/**
 * The client for /v1/chat. Same-origin: the engine serves this bundle, so there is
 * no proxy in the middle and no engine address in the browser.
 */

import { readSse } from "./sse";
import type { AguiEvent, HistoryTurn, Mode, SchemaTable } from "./types";

export async function* streamChat(
  question: string,
  mode: Mode,
  history: HistoryTurn[] = [],
  signal?: AbortSignal,
): AsyncGenerator<AguiEvent> {
  const response = await fetch("/v1/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, mode, history }),
    signal,
  });

  if (!response.ok || !response.body) {
    yield {
      type: "RUN_ERROR",
      message: `the engine returned ${response.status} ${response.statusText}`.trim(),
    };
    return;
  }

  for await (const data of readSse(response.body)) {
    let event: AguiEvent;
    try {
      event = JSON.parse(data) as AguiEvent;
    } catch {
      continue; // a frame we cannot parse is not a reason to drop the whole run
    }
    yield event;
  }
}

export async function fetchSchema(): Promise<SchemaTable[]> {
  const response = await fetch("/v1/schema");
  if (!response.ok) return [];
  const body = (await response.json()) as { tables?: SchemaTable[] };
  return body.tables ?? [];
}
