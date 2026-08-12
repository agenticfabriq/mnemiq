import { useEffect, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";

import { HistoryPanel } from "./components/HistoryPanel";
import { Mark } from "./components/Mark";
import { ModeSelector } from "./components/ModeSelector";
import { ScopePanel, useScope } from "./components/ScopePanel";
import { Thread } from "./components/Thread";
import { useWorkbench } from "./lib/runtime";

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

export default function App() {
  const { runtime, mode, setMode, ask, threads, activeId, selectThread, newChat, deleteThread } =
    useWorkbench();
  const scope = useScope();

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
