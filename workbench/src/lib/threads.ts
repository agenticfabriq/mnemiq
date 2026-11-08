/**
 * Conversation history.
 *
 * Threads live in localStorage: they survive a reload, which is what makes a history
 * panel worth having, without inventing a server-side store. Each identity's threads sit
 * in that identity's namespace (identity.ts), so a shared browser never shows one user
 * another's answers. History is still not a record of what the engine was asked.
 */

import { keyFor } from "./identity";
import type { Turn } from "./store";

const MAX_THREADS = 25;

export type Thread = {
  id: string;
  title: string;
  turns: Turn[];
  createdAt: number;
};

export const uid = (): string =>
  globalThis.crypto?.randomUUID?.() ??
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

export const emptyThread = (): Thread => ({
  id: uid(),
  title: "New chat",
  turns: [],
  createdAt: Date.now(),
});

/** The first question asked, which is what the reader recognises it by. */
export function titleFor(thread: Thread): string {
  const first = thread.turns.find((turn) => turn.role === "user");
  if (!first?.text) return "New chat";
  const line = first.text.trim().replace(/\s+/g, " ");
  return line.length > 48 ? `${line.slice(0, 47)}…` : line;
}

export const isBlank = (thread: Thread): boolean => thread.turns.length === 0;

/** `ns` null: a user with no namespace (identity.ts), for whom nothing is kept. */
export function load(ns: string | null): Thread[] {
  if (ns === null) return [];
  try {
    const raw = globalThis.localStorage?.getItem(keyFor(ns, "threads"));
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (t): t is Thread =>
        !!t && typeof t === "object" && typeof (t as Thread).id === "string" &&
        Array.isArray((t as Thread).turns),
    );
  } catch {
    return []; // unreadable history is not a reason to fail to start
  }
}

export function save(ns: string | null, threads: Thread[]): void {
  if (ns === null) return;
  const keep = threads.filter((t) => !isBlank(t)).slice(0, MAX_THREADS);
  for (let attempt = keep.length; attempt >= 0; attempt--) {
    try {
      globalThis.localStorage?.setItem(keyFor(ns, "threads"), JSON.stringify(keep.slice(0, attempt)));
      return;
    } catch {
      // Over quota: a result preview can be large, so drop the oldest and retry.
    }
  }
}
