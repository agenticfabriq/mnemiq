/**
 * The statement, collapsed by default; the answer comes first.
 *
 * It is the statement as ASKED (M51): under a row or column policy the executed one names the
 * policy's own tables and literals, and stays in the audit trace. When `narrowed` says the policy
 * changed something, the header says so and a note says the statement that ran is not shown;
 * otherwise nothing was rewritten (every rewrite records a narrowing) and it is the one that ran.
 * What was narrowed is named by the engine at the end of the answer's text (M155), so it is not
 * repeated here.
 */

import { useState } from "react";

import type { Narrowed } from "../lib/types";

export function SqlBlock({ sql, narrowed }: { sql: string; narrowed?: Narrowed[] | null }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  // Applied when a row filter or a masked column name touched any object: every rewrite sets one.
  const applied = (narrowed ?? []).some((n) => n.rows || n.columns);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(sql);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="border border-rule">
      <div className="flex items-center justify-between border-b border-rule bg-sunken">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="label flex items-center gap-1.5 px-2.5 py-1.5 hover:text-ink"
        >
          <span aria-hidden="true">{open ? "▾" : "▸"}</span>
          {applied ? "SQL as asked" : "Executed SQL"}
        </button>
        {open && (
          <button
            type="button"
            onClick={copy}
            className="label px-2.5 py-1.5 hover:text-ink"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        )}
      </div>
      {applied && (
        // Outside the toggle: that the statement shown is not the one that ran is part of the
        // answer, not a detail to open.
        <p className={`meta bg-brass-wash px-2.5 py-2 ${open ? "border-b border-rule" : ""}`}>
          The statement that ran also applied the policy, which is not shown here.
        </p>
      )}
      {open && (
        // Wrap rather than scroll: a statement is read, not scrubbed sideways, and a
        // horizontal scrollbar hides the tail of exactly the thing on trial here.
        <pre className="px-2.5 py-2 text-[12px] leading-relaxed break-words whitespace-pre-wrap">
          <code>{sql}</code>
        </pre>
      )}
    </div>
  );
}
