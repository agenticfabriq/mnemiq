/**
 * What this identity may see. The list comes from /v1/schema, which is access-scoped
 * server-side -- a table missing here is a table the engine will also refuse to read.
 *
 * It sits on the right and hides, because it is reference material: useful while you
 * work out what to ask, noise once you know.
 */

import { useEffect, useState } from "react";

import { fetchScope, type Scope } from "../lib/transport";
import type { SchemaTable } from "../lib/types";

const NOTHING: Scope = { tables: [], starters: [] };

/** One fetch for both, because the engine scopes them together (M32). `null` means the
 *  answer has not arrived yet, which the panel renders differently from an empty scope. */
export function useScope() {
  const [scope, setScope] = useState<Scope | null>(null);
  useEffect(() => {
    let live = true;
    fetchScope()
      .then((next) => live && setScope(next))
      .catch(() => live && setScope(NOTHING));
    return () => {
      live = false;
    };
  }, []);
  return scope;
}

export function ScopePanel({
  tables,
  onHide,
}: {
  tables: SchemaTable[] | null;
  onHide: () => void;
}) {
  const [filter, setFilter] = useState("");
  const needle = filter.trim().toLowerCase();
  const shown = (tables ?? []).filter((t) => t.object_id.toLowerCase().includes(needle));

  return (
    <aside className="hidden w-60 shrink-0 flex-col border-l border-rule lg:flex">
      <div className="flex items-center justify-between border-b border-rule px-3 py-2.5">
        <div>
          <h2 className="label">In scope</h2>
          <p className="tabular mt-0.5 text-[12px]">
            {tables === null ? "…" : `${tables.length} tables`}
          </p>
        </div>
        <button
          type="button"
          onClick={onHide}
          aria-label="Hide the table list"
          className="label px-1.5 py-1 hover:text-ink"
        >
          ✕
        </button>
      </div>

      {(tables?.length ?? 0) > 12 && (
        <div className="border-b border-rule px-3 py-1.5">
          <input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Filter"
            aria-label="Filter tables"
            className="w-full bg-transparent py-0.5 text-[12px] placeholder:text-graphite focus:outline-none"
          />
        </div>
      )}

      <ul className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
        {shown.map((table) => (
          <li
            key={table.object_id}
            title={table.object_id}
            className="truncate py-[1px] text-[11px] text-graphite/85"
          >
            {table.object_id}
          </li>
        ))}
      </ul>

      {tables?.length === 0 && (
        <p className="px-3 pb-3 text-[12px] text-graphite">
          No tables are granted to this identity. Access is fail-closed: set a policy
          and grant objects to a role.
        </p>
      )}
    </aside>
  );
}
