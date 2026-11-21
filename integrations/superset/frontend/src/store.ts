// The panel's store for the life of the page: the conversation and the running build's id, read with
// `useSyncExternalStore`. It lives in module memory only, never in browser storage, and never holds a measure's
// value (the measures section keeps those in its own state, read again each time it opens). rc2 unmounts the panel on
// every close (`AppContent` renders `ChatPanelHost` only while the panel is open), so what must outlive a close lives
// here.
import type { BuildRecord } from "./build";
import type { ChangeRecord } from "./change";
import type { Turn } from "./shared";

export type PanelState = {
  turns: Turn[];
  /** The build this page is driving, while it runs. */
  runningBuild: string | null;
  /** Whether a goal is being planned on this page (before its build runs). */
  planning: boolean;
  /** The build the panel shows: the latest started or finished on this page. */
  build: BuildRecord | null;
  /** The change by chat the panel shows: the latest sent on this page. */
  change: ChangeRecord | null;
};

const EMPTY: PanelState = { turns: [], runningBuild: null, planning: false, build: null, change: null };

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
