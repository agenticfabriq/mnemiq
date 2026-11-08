import { useCallback, useEffect, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";

import { HistoryPanel } from "./components/HistoryPanel";
import { Mark } from "./components/Mark";
import { ModeSelector } from "./components/ModeSelector";
import { ScopePanel, useScope } from "./components/ScopePanel";
import { Thread } from "./components/Thread";
import { enter, mayReload, namespaceOf, signOut } from "./lib/identity";
import { useWorkbench } from "./lib/runtime";
import { fetchWhoami, NOTICE, Refused, type Refusal } from "./lib/transport";

type Theme = "system" | "light" | "dark";

const SCOPE_KEY = "mnemiq.scope-open.v1";

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
  }, [theme]);

  const next: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
  return (
    <button
      type="button"
      onClick={() => setTheme(next[theme])}
      className="label border border-rule px-2 py-1 hover:text-ink"
      aria-label={`Theme: ${theme}. Switch to ${next[theme]}.`}
    >
      {theme}
    </button>
  );
}

type Notice = Exclude<Refusal, "signed-out">;

/** What the page shows before, or instead of, a user's workbench. */
type Session =
  | { state: "resolving" }
  | { state: "ready"; ns: string | null }
  | { state: "signed-out"; prompt: boolean }
  | { state: "refused"; refusal: Notice };

const RELOAD = () => globalThis.location.reload();

function SessionNotice({
  message,
  action,
}: {
  message: string;
  action?: { label: string; onClick: () => void };
}) {
  return (
    <main className="flex h-full items-center justify-center px-5">
      <section className="flex max-w-sm flex-col items-start gap-3">
        <span className="flex items-center gap-2">
          <Mark size={18} />
          <span className="text-[14px] font-medium tracking-tight">mnemiq</span>
        </span>
        <p className="text-[13px]">{message}</p>
        {action && (
          <button
            type="button"
            onClick={action.onClick}
            className="label border border-rule px-2.5 py-1.5 hover:border-brass hover:text-brass"
          >
            {action.label}
          </button>
        )}
      </section>
    </main>
  );
}

/**
 * Nothing of anyone's is rendered until the engine says who is asking: the namespace, and the
 * purge of every other one, come first. `reload` is a prop so a test can watch it.
 */
export default function App({ reload = RELOAD }: { reload?: () => void }) {
  const [session, setSession] = useState<Session>({ state: "resolving" });
  // A refusal once the workbench is up (the schema's): said above it, the threads kept.
  const [notice, setNotice] = useState<Notice | null>(null);

  const refused = useCallback(
    (refusal: Refusal) => {
      if (refusal !== "signed-out") {
        setNotice(refusal);
        return;
      }
      signOut();
      // Reloading goes back through the sign-in proxy, which signs the user in again. A proxy
      // that cannot would send every load straight back here, so a second reload within the
      // guard window asks the user instead.
      if (mayReload()) {
        setSession({ state: "signed-out", prompt: false });
        reload();
      } else {
        setSession({ state: "signed-out", prompt: true });
      }
    },
    [reload],
  );

  useEffect(() => {
    let live = true;
    fetchWhoami()
      .then(namespaceOf)
      .then((ns) => {
        if (!live) return;
        enter(ns);
        setSession({ state: "ready", ns });
      })
      .catch((error: unknown) => {
        if (!live) return;
        const refusal = error instanceof Refused ? error.refusal : "unavailable";
        if (refusal === "signed-out") refused(refusal);
        else setSession({ state: "refused", refusal });
      });
    return () => {
      live = false;
    };
  }, [refused]);

  if (session.state === "ready") {
    return <Workbench key={session.ns ?? ""} ns={session.ns} notice={notice} onRefused={refused} />;
  }
  if (session.state === "refused") return <SessionNotice message={NOTICE[session.refusal]} />;
  if (session.state === "signed-out" && session.prompt) {
    return (
      <SessionNotice
        message="Your session has ended. Sign in again to continue."
        action={{ label: "Sign in again", onClick: reload }}
      />
    );
  }
  return null; // resolving, or reloading
}

function Workbench({
  ns,
  notice,
  onRefused,
}: {
  ns: string | null;
  notice: Notice | null;
  onRefused: (refusal: Refusal) => void;
}) {
  const { runtime, mode, setMode, ask, threads, activeId, selectThread, newChat, deleteThread } =
    useWorkbench(ns, onRefused);
  const scope = useScope(onRefused);

  const [scopeOpen, setScopeOpen] = useState<boolean>(
    () => globalThis.localStorage?.getItem(SCOPE_KEY) !== "0",
  );
  useEffect(() => {
    globalThis.localStorage?.setItem(SCOPE_KEY, scopeOpen ? "1" : "0");
  }, [scopeOpen]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-full flex-col">
        <header className="flex items-center gap-4 border-b border-rule bg-surface px-5 py-2.5">
          <span className="flex items-center gap-2">
            <Mark size={18} />
            <span className="text-[14px] font-medium tracking-tight">mnemiq</span>
          </span>
          <span className="label hidden md:inline">workbench</span>
          <div className="ml-auto flex items-center gap-3">
            <ModeSelector mode={mode} onChange={setMode} />
            <button
              type="button"
              onClick={() => setScopeOpen((open) => !open)}
              aria-pressed={scopeOpen}
              className={`label hidden border px-2 py-1 lg:inline-block ${
                scopeOpen ? "border-rule text-ink" : "border-rule hover:text-ink"
              }`}
            >
              {scope === null ? "Tables" : `${scope.tables.length} tables`}
            </button>
            <ThemeToggle />
          </div>
        </header>

        {notice && (
          <p role="status" className="border-b border-rule bg-crimson-wash px-5 py-1.5 text-[12px] text-crimson">
            {NOTICE[notice]}
          </p>
        )}

        <div className="flex min-h-0 flex-1">
          <HistoryPanel
            threads={threads}
            activeId={activeId}
            onSelect={selectThread}
            onNew={newChat}
            onDelete={deleteThread}
          />
          <main className="min-w-0 flex-1">
            <Thread starters={scope?.starters ?? []} onPick={ask} />
          </main>
          {scopeOpen && <ScopePanel tables={scope?.tables ?? null} onHide={() => setScopeOpen(false)} />}
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}
