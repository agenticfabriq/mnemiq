/**
 * The assistant-ui binding. We own the state; assistant-ui owns the thread mechanics.
 *
 * The answer payload travels as a `data` message part named `mnemiq.turn`, which
 * MessagePrimitive.Parts dispatches by name to our renderer. The alternative --
 * bindExternalStoreMessage / getExternalStoreMessages -- is marked deprecated at
 * 0.15.4 and its generic is caller-asserted rather than checked.
 */

import { useCallback, useRef, useState } from "react";
import {
  useExternalStoreRuntime,
  type AppendMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";

import { streamChat } from "./transport";
import { isRunning, reduce, userTurn, type Turn } from "./store";
import { DEFAULT_MODE, type Mode } from "./types";

export const TURN_PART = "mnemiq.turn";

/** assistant-ui hands us structured parts; the composer only ever produces text. */
function textOf(message: AppendMessage): string {
  return message.content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("\n\n");
}

function convertMessage(turn: Turn): ThreadMessageLike {
  if (turn.role === "user") {
    return { id: turn.id, role: "user", content: [{ type: "text", text: turn.text }] };
  }
  // A refusal states its reason inside the deferral card, which also says what to do
  // next. Keeping the prose part as well would print that sentence twice.
  const refused = Boolean(turn.answer?.deferred || turn.answer?.failed);
  return {
    id: turn.id,
    role: "assistant",
    status:
      turn.status === "running"
        ? { type: "running" }
        : turn.status === "error"
          ? { type: "incomplete", reason: "error" }
          : { type: "complete", reason: "stop" },
    content: [
      ...(turn.text && !refused ? [{ type: "text" as const, text: turn.text }] : []),
      { type: "data" as const, name: TURN_PART, data: turn },
    ],
  };
}

export function useWorkbench() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [mode, setMode] = useState<Mode>(DEFAULT_MODE);
  // The composer reads the mode at send time, so a mid-run change never retargets
  // the run already in flight.
  const modeRef = useRef(mode);
  modeRef.current = mode;

  const ask = useCallback(async (question: string) => {
    setTurns((current) => [...current, userTurn(question)]);
    for await (const event of streamChat(question, modeRef.current)) {
      setTurns((current) => reduce(current, event));
    }
  }, []);

  const runtime = useExternalStoreRuntime<Turn>({
    messages: turns,
    isRunning: isRunning(turns),
    convertMessage,
    onNew: async (message: AppendMessage) => {
      const question = textOf(message).trim();
      if (question) await ask(question);
    },
  });

  return { runtime, mode, setMode, ask, turns };
}
