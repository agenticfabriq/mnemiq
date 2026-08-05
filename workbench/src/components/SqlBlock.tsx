/** The statement that actually ran. Collapsed by default; the answer comes first. */

import { useState } from "react";

export function SqlBlock({ sql }: { sql: string }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

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
          Executed SQL
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
