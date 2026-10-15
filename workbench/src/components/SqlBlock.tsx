/**
 * The statement, collapsed by default; the answer comes first.
 *
 * It is the statement as ASKED (M51): under a row or column policy the executed one names the
 * policy's own tables and literals, and stays in the audit trace. When `narrowed` says the policy
 * changed something, the header says so and a note names the objects; otherwise nothing was
 * rewritten (every rewrite records a narrowing) and the statement shown is the one that ran.
 */

import { useState } from "react";

import type { Narrowed } from "../lib/types";

export function SqlBlock({ sql, narrowed }: { sql: string; narrowed?: Narrowed[] | null }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const withheld = (narrowed ?? []).filter((n) => n.rows).map((n) => n.object);
  const masked = (narrowed ?? []).filter((n) => n.columns).map((n) => n.object);
  const applied = withheld.length > 0 || masked.length > 0;

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
        // Outside the toggle: what the policy did is part of the answer, not a detail to open.
        // "Referenced on", not "masked on": the flag can name a table whose column the caller
        // never read, where ownership does not resolve.
        <div className={`bg-brass-wash px-2.5 py-2 text-[12px] ${open ? "border-b border-rule" : ""}`}>
          {withheld.length > 0 && <p>Rows withheld from: {withheld.join(", ")}.</p>}
          {masked.length > 0 && <p>Masked columns referenced on: {masked.join(", ")}.</p>}
          <p className="meta mt-1">
            The statement that ran also applied the policy, which is not shown here.
          </p>
        </div>
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
