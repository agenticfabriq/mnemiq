import { useCallback, useEffect, useRef, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";

import { HistoryPanel } from "./components/HistoryPanel";
import { Mark } from "./components/Mark";
import { ModeSelector } from "./components/ModeSelector";
import { ScopePanel, useScope } from "./components/ScopePanel";
import { Thread } from "./components/Thread";
import { enter, MARKER, mayReload, namespaceOf, signOut, signOutHref } from "./lib/identity";
import { useWorkbench } from "./lib/runtime";
import { fetchWhoami, NOTICE, Refused, type Refusal } from "./lib/transport";
import type { Identity, Whoami } from "./lib/types";

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
  // `ns` is the identity's namespace, `shelf` where its items are kept (`enter`).
  | { state: "ready"; ns: string | null; shelf: string | null; verified: boolean; endSession: string | null }
  | { state: "signed-out"; prompt: boolean }
  | { state: "leaving"; ns: string | null; endSession: string | null }
  // `endSession`: where a Sign out from the notice ends, as the refusing 403 named it.
  | { state: "refused"; refusal: Notice; endSession: string | null };

const RELOAD = () => globalThis.location.reload();
const NAVIGATE = (url: string) => globalThis.location.assign(url);

/** Where a Sign out sends the browser (`signOutHref`), returning to the page signed out from. */
const signOutTo = (endSession: string | null): string =>
  signOutHref(endSession, globalThis.location.origin, `${globalThis.location.pathname}${globalThis.location.search}`);

/**
 * The identity this tab holds threads for, as of its last check. The proxy's session is the
 * browser's, not the tab's: someone else can sign in through another tab while this one stays
 * open, and its next request then carries their credentials. So a question is preceded by a
 * fresh check when the tab has been hidden since, or the last check is older than `IDLE_MS`.
 * The raw id stays in memory here and is never stored.
 */
type Checked = { who: string; ns: string | null; at: number; hidden: boolean };

const IDLE_MS = 60_000;

const whoKey = (who: Identity | null): string =>
  who === null ? "" : JSON.stringify([who.tenant_id, who.principal_id, who.verified]);

/** A verified user on a page that cannot hash their identity gets no saved threads. Said,
 *  so that history vanishing on reload is not mistaken for a fault. */
const UNSAVED =
  "Saved threads are off on this page: keeping each user's threads apart needs a secure context, such as HTTPS.";

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
 * purge of every other one, come first. `reload` and `navigate` are props so a test can watch
 * them.
 */
export default function App({
  reload = RELOAD,
  navigate = NAVIGATE,
}: {
  reload?: () => void;
  navigate?: (url: string) => void;
}) {
  const [session, setSession] = useState<Session>({ state: "resolving" });
  // A refusal once the workbench is up (the schema's): said above it, the threads kept.
  const [notice, setNotice] = useState<Notice | null>(null);
  const checked = useRef<Checked | null>(null);
  // Bumped to ask the engine again who is asking.
  const [attempt, setAttempt] = useState(0);

  /** The browser now answers for someone else: drop everything this tab holds (unmounting the
   *  workbench cancels its questions in flight) and ask again who is asking. */
  const changed = useCallback(() => {
    checked.current = null;
    setNotice(null);
    setSession({ state: "resolving" });
    setAttempt((n) => n + 1);
  }, []);

  const refused = useCallback(
    (refusal: Refusal) => {
      if (refusal !== "signed-out") {
        setNotice(refusal);
        return;
      }
      checked.current = null;
      // An expired session clears the screen and this tab's memory, and keeps the saved
      // threads. The same user usually signs straight back in, and a refusal can arrive late,
      // from a tab opened before someone else signed in, where a purge would erase the new
      // user's history. A different user's threads are purged when that user loads (`enter`);
      // the Sign out control clears the user's own.
      //
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
      .then(async (who) => ({ who, ns: await namespaceOf(who) }))
      .then(({ who, ns }) => {
        if (!live) return;
        const shelf = enter(ns);
        checked.current = { who: whoKey(who), ns, at: Date.now(), hidden: false };
        setSession({
          state: "ready",
          ns,
          shelf,
          verified: who !== null && who.verified,
          endSession: who?.end_session_url ?? null,
        });
      })
      .catch((error: unknown) => {
        if (!live) return;
        const why = error instanceof Refused ? error : new Refused("unavailable");
        if (why.refusal === "signed-out") refused(why.refusal);
        else setSession({ state: "refused", refusal: why.refusal, endSession: why.endSession });
      });
    return () => {
      live = false;
    };
  }, [refused, attempt]);

  // Another tab that loads as someone else writes the identity marker, and the browser tells
  // every other tab. (A cleared storage reads as a change too.)
  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.storageArea && event.storageArea !== globalThis.localStorage) return;
      if (event.key !== null && event.key !== MARKER) return;
      const mine = checked.current;
      if (mine && event.newValue !== mine.ns) changed();
    };
    globalThis.addEventListener("storage", onStorage);
    return () => globalThis.removeEventListener("storage", onStorage);
  }, [changed]);

  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState === "hidden" && checked.current) checked.current.hidden = true;
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, []);

  /** Whether the user is still the one this tab holds threads for; asked before a question. */
  const confirm = useCallback(async (): Promise<boolean> => {
    const mine = checked.current;
    if (!mine) return false;
    if (!mine.hidden && Date.now() - mine.at < IDLE_MS) return true;
    let who: Whoami | null;
    try {
      who = await fetchWhoami();
    } catch (error) {
      // Nothing is sent without a confirmed identity, so a failed check sends nothing.
      if (checked.current === mine) refused(error instanceof Refused ? error.refusal : "unavailable");
      return false;
    }
    if (checked.current !== mine) return false; // the page moved on while asking
    if (whoKey(who) !== mine.who) {
      changed();
      return false;
    }
    mine.at = Date.now();
    mine.hidden = false;
    return true;
  }, [changed, refused]);

  const signOutNow = useCallback(() => {
    checked.current = null;
    setSession((current) =>
      current.state === "ready" ? { state: "leaving", ns: current.ns, endSession: current.endSession } : current,
    );
  }, []);

  useEffect(() => {
    if (session.state !== "leaving") return;
    // Run once the workbench is gone, so nothing is left to write the threads back.
    signOut(session.ns);
    navigate(signOutTo(session.endSession));
  }, [session, navigate]);

  if (session.state === "ready") {
    return (
      <Workbench
        key={session.shelf ?? ""}
        shelf={session.shelf}
        notice={notice}
        onRefused={refused}
        confirm={confirm}
        onSignOut={session.verified ? signOutNow : null}
      />
    );
  }
  if (session.state === "refused") {
    const { refusal, endSession } = session;
    // Refused as this sign-in, signing in again as the same user changes nothing: the way on is
    // another account. Nothing is purged, since a 403 does not say who is asking, so nothing this
    // browser holds is known to be theirs. An outage is not about the sign-in, and offers nothing.
    const action =
      refusal === "unavailable" ? undefined : { label: "Sign out", onClick: () => navigate(signOutTo(endSession)) };
    return <SessionNotice message={NOTICE[refusal]} action={action} />;
  }
  if (session.state === "signed-out" && session.prompt) {
    return (
      <SessionNotice
        message="Your session has ended. Sign in again to continue."
        action={{ label: "Sign in again", onClick: reload }}
      />
    );
  }
  return null; // resolving, reloading, or leaving
}

/** `shelf`: where this identity's threads and mode are kept (`enter`); null keeps nothing.
 *  `onSignOut`: present when there is a sign-in to end, that is, the identity is verified. */
function Workbench({
  shelf,
  notice,
  onRefused,
  confirm,
  onSignOut,
}: {
  shelf: string | null;
  notice: Notice | null;
  onRefused: (refusal: Refusal) => void;
  confirm: () => Promise<boolean>;
  onSignOut: (() => void) | null;
}) {
  const { runtime, mode, setMode, ask, threads, activeId, selectThread, newChat, deleteThread } =
    useWorkbench(shelf, onRefused, confirm);
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
            {onSignOut && (
              <button
                type="button"
                onClick={onSignOut}
                className="label border border-rule px-2 py-1 hover:text-ink"
              >
                Sign out
              </button>
            )}
          </div>
        </header>

        {notice && (
          <p role="status" className="border-b border-rule bg-crimson-wash px-5 py-1.5 text-[12px] text-crimson">
            {NOTICE[notice]}
          </p>
        )}
        {shelf === null && (
          <p role="status" className="border-b border-rule px-5 py-1.5 text-[12px] text-graphite">
            {UNSAVED}
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
