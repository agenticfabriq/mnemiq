/**
 * Past conversations, newest first. Stored in this browser -- see lib/threads.ts for
 * what that does and does not promise.
 */

import type { Thread } from "../lib/threads";

export function HistoryPanel({
  threads,
  activeId,
  onSelect,
  onNew,
  onDelete,
}: {
  threads: Thread[];
  activeId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}) {
  return (
    <aside className="hidden w-56 shrink-0 flex-col border-r border-rule md:flex">
      <div className="border-b border-rule px-3 py-2.5">
        <button
          type="button"
          onClick={onNew}
          className="label w-full border border-rule px-2 py-1.5 hover:border-brass hover:text-brass"
        >
          + New chat
        </button>
      </div>

      <nav aria-label="History" className="min-h-0 flex-1 overflow-y-auto py-1.5">
        {threads.map((thread) => {
          const current = thread.id === activeId;
          return (
            <div
              key={thread.id}
              className={`group flex items-center gap-1 px-1.5 ${
                current ? "bg-sunken" : ""
              }`}
            >
              <button
                type="button"
                onClick={() => onSelect(thread.id)}
                aria-current={current ? "true" : undefined}
                title={thread.title}
                className={`min-w-0 flex-1 truncate px-1.5 py-1.5 text-left text-[12px] ${
                  current ? "text-ink" : "text-graphite hover:text-ink"
                }`}
              >
                {thread.title}
              </button>
              <button
                type="button"
                onClick={() => onDelete(thread.id)}
                aria-label={`Delete ${thread.title}`}
                className="label px-1.5 py-1 opacity-0 group-hover:opacity-100 hover:text-crimson focus-visible:opacity-100"
              >
                ✕
              </button>
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
