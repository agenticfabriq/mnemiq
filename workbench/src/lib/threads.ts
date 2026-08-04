/**
 * Conversation history.
 *
 * Threads live in localStorage: they survive a reload, which is what makes a history
 * panel worth having, without inventing a server-side store or a per-identity
 * schema. The consequence is worth stating plainly -- history is per browser, not
 * per principal, and it is not a record of what the engine was asked.
 */

import type { Turn } from "./store";

const KEY = "mnemiq.threads.v1";
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

export function load(): Thread[] {
  try {
    const raw = globalThis.localStorage?.getItem(KEY);
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

export function save(threads: Thread[]): void {
  const keep = threads.filter((t) => !isBlank(t)).slice(0, MAX_THREADS);
  for (let attempt = keep.length; attempt >= 0; attempt--) {
    try {
      globalThis.localStorage?.setItem(KEY, JSON.stringify(keep.slice(0, attempt)));
      return;
    } catch {
      // Over quota: a result preview can be large, so drop the oldest and retry.
    }
  }
}
