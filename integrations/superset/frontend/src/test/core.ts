// A fake of the part of Superset's `@apache-superset/core` the panel calls, for the unit tests (aliased in
// vitest.config.ts). It behaves as rc2's host does where the panel can tell (`src/core/chat/index.ts`,
// `src/core/navigation/index.ts`): `open` and `close` fire their events only when the state changes, and the display
// mode event fires on a change. `host` lets a test read what the panel asked of it and drive the page.
import type { ComponentType } from "react";

type Listener<T> = (e: T) => unknown;

function emitter<T>() {
  const listeners = new Set<Listener<T>>();
  const event = (listener: Listener<T>) => {
    listeners.add(listener);
    return { dispose: () => void listeners.delete(listener) };
  };
  return { event, fire: (e: T) => [...listeners].forEach((listener) => listener(e)), clear: () => listeners.clear() };
}

type DisplayMode = "floating" | "panel";
type Page = "dashboard" | "dashboard_list" | "explore" | "chart_list" | "sqllab" | "query_history" | "saved_queries"
  | "dataset" | "dataset_list" | "home";

const opened = emitter<void>();
const closed = emitter<void>();
const modeChanged = emitter<DisplayMode>();
const pageChanged = emitter<Page>();

export const host = {
  registered: [] as { chat: { id: string; name: string; description?: string }; trigger: ComponentType; panel: ComponentType }[],
  open: false,
  mode: "floating" as DisplayMode,
  page: "home" as Page,
  csrf: "csrf-for-the-session" as string | undefined,
  /** What the panel asked of the host, in order: `open`, `close`, `setDisplayMode:<mode>`. */
  calls: [] as string[],
  /** Moves the page as rc2's router does: the event fires only when the derived page changes. */
  navigate(page: Page) {
    if (page === host.page) return;
    host.page = page;
    pageChanged.fire(page);
  },
  reset() {
    host.registered = [];
    host.open = false;
    host.mode = "floating";
    host.page = "home";
    host.csrf = "csrf-for-the-session";
    host.calls = [];
    for (const e of [opened, closed, modeChanged, pageChanged]) e.clear();
  },
};

export const chat = {
  registerChat(chat: { id: string; name: string; description?: string }, trigger: ComponentType, panel: ComponentType) {
    const entry = { chat, trigger, panel };
    host.registered.push(entry);
    return { dispose: () => void (host.registered = host.registered.filter((e) => e !== entry)) };
  },
  open() {
    host.calls.push("open");
    if (host.open) return;
    host.open = true;
    opened.fire();
  },
  close() {
    host.calls.push("close");
    if (!host.open) return;
    host.open = false;
    closed.fire();
  },
  isOpen: () => host.open,
  onDidOpen: opened.event,
  onDidClose: closed.event,
  getDisplayMode: () => host.mode,
  setDisplayMode(mode: DisplayMode) {
    host.calls.push(`setDisplayMode:${mode}`);
    if (mode === host.mode) return;
    host.mode = mode;
    modeChanged.fire(mode);
  },
  onDidChangeDisplayMode: modeChanged.event,
};

export const navigation = {
  getPage: () => host.page,
  onDidChangePage: pageChanged.event,
};

export const authentication = {
  getCSRFToken: async () => host.csrf,
};
