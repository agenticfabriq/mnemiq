/**
 * What this identity may see. The list comes from /v1/schema, which is access-scoped
 * server-side -- a table missing here is a table the engine will also refuse to read.
 */

import { useEffect, useState } from "react";

import { fetchSchema } from "../lib/transport";
import type { SchemaTable } from "../lib/types";

export function ScopePanel() {
  const [tables, setTables] = useState<SchemaTable[] | null>(null);

  useEffect(() => {
    let live = true;
    fetchSchema()
      .then((rows) => live && setTables(rows))
      .catch(() => live && setTables([]));
    return () => {
      live = false;
    };
  }, []);

  return (
    <aside className="hidden w-56 shrink-0 flex-col overflow-y-auto border-r border-rule lg:flex">
      <div className="border-b border-rule px-4 py-3">
        <h2 className="label">In scope</h2>
        <p className="mt-1 text-[12px] tabular">
          {tables === null ? "…" : `${tables.length} tables`}
        </p>
      </div>
      <ul className="flex flex-col px-4 py-2">
        {(tables ?? []).map((table) => (
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
        <p className="px-4 pb-3 text-[12px] text-graphite">
          No tables are granted to this identity. Access is fail-closed: set a policy
          and grant objects to a role.
        </p>
      )}
    </aside>
  );
}
