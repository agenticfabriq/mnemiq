/**
 * AG-UI events -> the message list.
 *
 * Pure on purpose: no React, no network. The reduction is where a streaming chat UI
 * actually goes wrong, so it is the part that gets tested directly.
 *
 * An event that changes nothing returns the SAME array reference. assistant-ui's
 * external store short-circuits re-conversion on `oldStore.messages === messages`,
 * so identity here is a correctness-adjacent performance contract, not a nicety.
 */

import type { AguiEvent, AnswerPayload } from "./types";

export type TurnStatus = "running" | "complete" | "error";

export type Turn = {
  id: string;
  role: "user" | "assistant";
  text: string;
  /** Present once the CUSTOM frame lands; absent on a stream that errored early. */
  answer?: AnswerPayload;
  error?: string;
  status: TurnStatus;
};

export const ANSWER_EVENT = "mnemiq.answer";

let seq = 0;
const nextId = (prefix: string) => `${prefix}-${++seq}`;

export function userTurn(text: string): Turn {
  return { id: nextId("u"), role: "user", text, status: "complete" };
}

/** Replace the last assistant turn, if there is one; otherwise leave the list alone. */
function patchLast(turns: Turn[], patch: (turn: Turn) => Turn): Turn[] {
  for (let i = turns.length - 1; i >= 0; i--) {
    const turn = turns[i]!;
    if (turn.role !== "assistant") continue;
    const next = patch(turn);
    if (next === turn) return turns;
    const copy = turns.slice();
    copy[i] = next;
    return copy;
  }
  return turns;
}

export function reduce(turns: Turn[], event: AguiEvent): Turn[] {
  switch (event.type) {
    case "RUN_STARTED":
      return [
        ...turns,
        {
          id: "runId" in event && event.runId ? event.runId : nextId("a"),
          role: "assistant",
          text: "",
          status: "running",
        },
      ];

    case "TEXT_MESSAGE_CONTENT": {
      const delta = "delta" in event ? String(event.delta ?? "") : "";
      if (!delta) return turns;
      return patchLast(turns, (t) => ({ ...t, text: t.text + delta }));
    }

    case "CUSTOM": {
      if (!("name" in event) || event.name !== ANSWER_EVENT) return turns;
      const answer = event.value as AnswerPayload;
      return patchLast(turns, (t) => ({ ...t, answer }));
    }

    case "RUN_FINISHED":
      return patchLast(turns, (t) =>
        t.status === "complete" ? t : { ...t, status: "complete" },
      );

    case "RUN_ERROR": {
      const message = "message" in event ? String(event.message) : "the run failed";
      return patchLast(turns, (t) => ({ ...t, status: "error", error: message }));
    }

    // TEXT_MESSAGE_START/END carry no state we keep, and a server that later emits
    // richer progress frames must not break a client that predates them.
    default:
      return turns;
  }
}

export const isRunning = (turns: Turn[]): boolean =>
  turns.some((t) => t.status === "running");
