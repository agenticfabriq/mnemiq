/**
 * The signature element: one status line per assistant turn, pinned to a rule whose
 * colour is the disposition. Read down the transcript and you read a column of
 * verdicts -- which is the most load-bearing fact about each turn.
 */

import type { Turn } from "../lib/store";
import {
  DISPOSITION_LABEL,
  disposition,
  duration,
  plural,
  type Disposition,
} from "../lib/verdict";

const RULE: Record<Disposition, string> = {
  answered: "bg-rule",
  declined: "bg-brass",
  failed: "bg-crimson",
  broken: "bg-crimson",
  working: "bg-graphite",
};

const TEXT: Record<Disposition, string> = {
  answered: "text-graphite",
  declined: "text-brass",
  failed: "text-crimson",
  broken: "text-crimson",
  working: "text-graphite",
};

export function VerdictStripe({ turn }: { turn: Turn }) {
  const state = disposition(turn);
  const answer = turn.answer;
  const facts: string[] = [];

  if (answer?.mode) facts.push(answer.mode);
  const total = duration(answer?.timing?.["total_ms"]);
  if (total) facts.push(total);
  if (answer?.tables_used?.length) {
    facts.push(plural(answer.tables_used.length, "table"));
  }
  if (answer?.enrichment_version) facts.push(answer.enrichment_version);

  return (
    <div className="flex items-center gap-2.5">
      <span className={`h-3.5 w-[3px] shrink-0 ${RULE[state]}`} aria-hidden="true" />
      <span className={`label ${TEXT[state]} font-medium`}>
        {DISPOSITION_LABEL[state]}
      </span>
      {/* `meta`, not `label`: these are the engine's own strings. */}
      {facts.length > 0 && (
        <span className="meta tabular truncate">{facts.join("  ·  ")}</span>
      )}
    </div>
  );
}
