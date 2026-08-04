import { useEffect, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";

import { ModeSelector } from "./components/ModeSelector";
import { ScopePanel } from "./components/ScopePanel";
import { Thread } from "./components/Thread";
import { useWorkbench } from "./lib/runtime";

type Theme = "system" | "light" | "dark";

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
  const { runtime, mode, setMode, ask } = useWorkbench();

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-full flex-col">
        <header className="flex items-center gap-4 border-b border-rule bg-surface px-5 py-2.5">
          <span className="text-[14px] font-medium tracking-tight">mnemiq</span>
          <span className="label hidden md:inline">workbench</span>
          <div className="ml-auto flex items-center gap-3">
            <ModeSelector mode={mode} onChange={setMode} />
            <ThemeToggle />
          </div>
        </header>

        <div className="flex min-h-0 flex-1">
          <ScopePanel />
          <main className="min-w-0 flex-1">
            <Thread onPick={ask} />
          </main>
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}
