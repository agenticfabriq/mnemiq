/**
 * The assistant-ui binding. We own the state; assistant-ui owns the thread mechanics.
 *
 * The answer payload travels as a `data` message part named `mnemiq.turn`, which
 * MessagePrimitive.Parts dispatches by name to our renderer. The alternative --
 * bindExternalStoreMessage / getExternalStoreMessages -- is marked deprecated at
 * 0.15.4 and its generic is caller-asserted rather than checked.
 *
 * Threads are ours too. assistant-ui's external-store thread-list adapter is
 * deprecated at this version, and switching threads is just handing the runtime a
 * different messages array -- not worth taking a churning API for.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useExternalStoreRuntime,
  type AppendMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";

import { historyFrom } from "./history";
import { streamChat } from "./transport";
import { isRunning, reduce, userTurn, type Turn } from "./store";
import { emptyThread, isBlank, load, save, titleFor, type Thread } from "./threads";
import { loadMode, saveMode } from "./mode";
import { type Mode } from "./types";

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
  const restored = useMemo(() => {
    const stored = load();
    return stored.length ? stored : [emptyThread()];
  }, []);

  const [threads, setThreads] = useState<Thread[]>(restored);
  const [activeId, setActiveId] = useState<string>(restored[0]!.id);
  // Restored, not defaulted: mode is what a question costs, and a reload that quietly
  // moves a reader from `deep` back to `thinking` changes the answer they get next.
  const [mode, setMode] = useState<Mode>(loadMode);

  // Read at send time so switching thread or mode mid-run never retargets a run
  // already in flight.
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const activeRef = useRef(activeId);
  activeRef.current = activeId;
  const threadsRef = useRef(threads);
  threadsRef.current = threads;

  useEffect(() => {
    saveMode(mode);
  }, [mode]);

  useEffect(() => {
    save(threads);
  }, [threads]);

  const active = threads.find((t) => t.id === activeId) ?? threads[0]!;

  const updateTurns = useCallback(
    (threadId: string, update: (turns: Turn[]) => Turn[]) => {
      setThreads((current) =>
        current.map((thread) => {
          if (thread.id !== threadId) return thread;
          const turns = update(thread.turns);
          if (turns === thread.turns) return thread;
          const next = { ...thread, turns };
          return { ...next, title: titleFor(next) };
        }),
      );
    },
    [],
  );

  const ask = useCallback(
    async (question: string) => {
      const threadId = activeRef.current;
      // Read before the user turn is appended: the history is what came BEFORE this ask.
      const before = threadsRef.current.find((t) => t.id === threadId)?.turns ?? [];
      const history = historyFrom(before);
      updateTurns(threadId, (current) => [...current, userTurn(question)]);
      for await (const event of streamChat(question, modeRef.current, history)) {
        updateTurns(threadId, (current) => reduce(current, event));
      }
    },
    [updateTurns],
  );

  const newChat = useCallback(() => {
    const blank = threadsRef.current.find(isBlank);
    if (blank) {
      setActiveId(blank.id);
      return;
    }
    const thread = emptyThread();
    setThreads((current) => [thread, ...current]);
    setActiveId(thread.id);
  }, []);

  const deleteThread = useCallback((id: string) => {
    const remaining = threadsRef.current.filter((thread) => thread.id !== id);
    const next = remaining.length > 0 ? remaining : [emptyThread()];
    setThreads(next);
    if (id === activeRef.current) setActiveId(next[0]!.id);
  }, []);

  const runtime = useExternalStoreRuntime<Turn>({
    messages: active.turns,
    isRunning: isRunning(active.turns),
    convertMessage,
    onNew: async (message: AppendMessage) => {
      const question = textOf(message).trim();
      if (question) await ask(question);
    },
  });

  return {
    runtime,
    mode,
    setMode,
    ask,
    threads,
    activeId: active.id,
    selectThread: setActiveId,
    newChat,
    deleteThread,
  };
}
