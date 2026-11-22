// The panel's store for the life of the page: the conversation on screen and the running build's id, read with
// `useSyncExternalStore`. It lives in module memory only, never in browser storage, and never holds a row of results
// or a measure's value (a reopened answer's table and the measures section keep theirs in their own component state).
// mnemiq keeps the conversation itself; this is what is on screen. rc2 unmounts the panel on every close
// (`AppContent` renders `ChatPanelHost` only while the panel is open), so what must outlive a close lives here.
import type { BuildRecord } from "./build";
import type { Message } from "./conversation";
import type { Turn } from "./shared";

export type PanelState = {
  /** The conversation on screen, by mnemiq's id; none until its first message is sent. */
  conversationId: string | null;
  title: string;
  /** Its messages as mnemiq last answered them. */
  messages: Message[];
  /** The message in flight: the person's turn and mnemiq's as its stream arrives, until mnemiq's copy is read. */
  turns: Turn[];
  sending: boolean;
  /** A sentence about the last confirm, decline or read that did not go through. */
  notice: string | null;
  /** The build this page is driving, while it runs. */
  runningBuild: string | null;
  /** Whether a goal is being planned on this page (before its build runs). */
  planning: boolean;
  /** The build the panel shows: the latest started or finished on this page. */
  build: BuildRecord | null;
};

const EMPTY: PanelState = { conversationId: null, title: "", messages: [], turns: [], sending: false, notice: null,
  runningBuild: null, planning: false, build: null };

let state: PanelState = EMPTY;
const listeners = new Set<() => void>();

export function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => void listeners.delete(listener);
}

export const getSnapshot = (): PanelState => state;

/** Replaces the state with `fn`'s answer; listeners hear of it only when it changed. */
export function update(fn: (current: PanelState) => PanelState): void {
  const next = fn(state);
  if (next === state) return;
  state = next;
  for (const listener of [...listeners]) listener();
}

/** For tests: the store as a fresh page has it. */
export function resetStore(): void {
  state = EMPTY;
  for (const listener of [...listeners]) listener();
}
