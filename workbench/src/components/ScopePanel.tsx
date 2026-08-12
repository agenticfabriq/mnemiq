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

/** The filter survives a reload, beside the panel's open state and the threads. */
const FILTER_KEY = "mnemiq.scope-filter.v1";

const loadFilter = (): string => {
  try {
    return globalThis.localStorage?.getItem(FILTER_KEY) ?? "";
  } catch {
    return "";
  }
};

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
  const [filter, setFilter] = useState(loadFilter);
  useEffect(() => {
    try {
      // Removed rather than stored empty, so "no filter" is the absence of a key and not a
      // value that has to be interpreted.
      if (filter) globalThis.localStorage?.setItem(FILTER_KEY, filter);
      else globalThis.localStorage?.removeItem(FILTER_KEY);
    } catch {
      // Storage denied or over quota: the session keeps the filter, only the memory is lost.
    }
  }, [filter]);

  const needle = filter.trim().toLowerCase();
  const filtering = needle !== "";
  const all = tables ?? [];
  const shown = all.filter((t) => t.object_id.toLowerCase().includes(needle));

  return (
    <aside className="hidden w-60 shrink-0 flex-col border-l border-rule lg:flex">
      <div className="flex items-center justify-between border-b border-rule px-3 py-2.5">
        <div>
          <h2 className="label">In scope</h2>
          <p className="tabular mt-0.5 text-[12px]">
            {/* `N of M` while filtering. The header read the TOTAL whatever the filter did, so a
                filtered panel claimed 18 tables while listing 4 -- harmless when the filter died
                on reload, a standing lie once it survives one, because the reader arrives with no
                memory of having typed it. */}
            {tables === null
              ? "…"
              : filtering
                ? `${shown.length} of ${all.length} tables`
                : `${all.length} tables`}
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

      {/* Above the threshold, OR whenever a filter is active: a filter stored against a wide
          scope and reloaded against a narrow one would otherwise apply invisibly, shortening the
          list with no control on screen to clear it. */}
      {((tables?.length ?? 0) > 12 || filtering) && (
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
