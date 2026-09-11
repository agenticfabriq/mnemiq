/**
 * What the engine decided, and what the reader should do about it.
 *
 * The copy for each reason comes from what the caller is expected to do next --
 * the axis `DeferralReason` is categorised on. A refusal is a verdict with an
 * action attached, not an apology.
 */

import type { Turn } from "./store";
import type { DeferralReason } from "./types";

export type Disposition = "answered" | "declined" | "failed" | "working" | "broken";

export function disposition(turn: Turn): Disposition {
  if (turn.status === "error") return "broken";
  if (!turn.answer) return turn.status === "running" ? "working" : "answered";
  if (turn.answer.failed) return "failed";
  if (turn.answer.deferred) return "declined";
  return "answered";
}

export const DISPOSITION_LABEL: Record<Disposition, string> = {
  answered: "Answered",
  declined: "Declined",
  // Three codes reach this disposition and only one of them is the source. It said
  // "Source failed" over a model-provider outage before a verifier outage could reach
  // it too; the card below names which one broke.
  failed: "Could not answer",
  working: "Working",
  broken: "Stream failed",
};

export const REASONS: Record<DeferralReason, { title: string; next: string }> = {
  authorization: {
    title: "Out of scope for this identity",
    next: "Request a grant for the tables this question needs.",
  },
  policy_unavailable: {
    title: "The access policy could not be read",
    next: "Page an operator. The engine refuses rather than guess at access.",
  },
  no_tables: {
    title: "Nothing retrieved for this question",
    next: "Rephrase it, or check that enrichment has run over this source.",
  },
  undefined_term: {
    // M35. Distinct from `unanswerable` because the caller's next move is different: the tables
    // CAN answer it and nobody has said what the term means, so rephrasing is the one thing that
    // will not work. Without this entry the card fell through to UNKNOWN and told the operator
    // the code was unrecognised -- in exactly the case the engine had just added a code for.
    title: "No certified definition for a term in the question",
    next: "Certify a definition for the term named above, then ask again. Rephrasing will not help — the engine can read the data and does not know what the term means.",
  },
  unanswerable: {
    title: "Not answerable from the tables in scope",
    next: "Ask about a column the data actually holds.",
  },
  invalid_query: {
    title: "No valid query within the budget",
    next: "Narrow the question, or retry in a higher mode.",
  },
  verification: {
    title: "Withdrawn by the verifier",
    next: "The engine answered, then declined to stand behind it. Narrow the question.",
  },
  disagreement: {
    title: "Candidates disagreed",
    next: "Too much divergence to pick one answer. Retry, or narrow the question.",
  },
  execution_failed: {
    title: "The source rejected every attempt",
    next: "Page an operator. This is a source failure, not a refusal.",
  },
  model_unavailable: {
    title: "The model provider did not respond",
    next: "An outage, not a judgement about your data. Try again.",
  },
  // Third code riding `failed: true`, and the third thing that can break. The engine ran the
  // query and could not get the answer CHECKED, which is why it is not `verification` -- that
  // one means a judge read the result and declined to stand behind it.
  verifier_unavailable: {
    title: "The answer could not be verified",
    // "no usable answer", not "did not answer": the engine sends this code both when the judge
    // endpoint was unreachable and when it replied with something no confidence could be read
    // out of. The body above says which, and saying "did not answer" over the second sends the
    // operator to check connectivity when the cause is a model or a token budget.
    next: "The query ran; the verifier gave no usable answer, so the result was withheld rather than returned unchecked. An outage, not a judgement about your data.",
  },
};

/** 412 ms / 8.5 s / 2 m 04 s -- always legible, never more precision than is useful. */
export function duration(ms: number | undefined): string | null {
  if (ms === undefined || Number.isNaN(ms)) return null;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const minutes = Math.floor(ms / 60_000);
  return `${minutes} m ${String(Math.round((ms % 60_000) / 1000)).padStart(2, "0")} s`;
}

export const plural = (n: number, one: string, many = `${one}s`) =>
  `${n} ${n === 1 ? one : many}`;

/**
 * What the mode spent, in words -- "1 attempt · no repair needed".
 *
 * `instant` and `thinking` run identical code whenever the first SQL is approved: the
 * corrector fires only on a decider refusal, and attempts 2 and 3 only after the database
 * rejects one. So the control looked inert on most questions while working exactly as
 * designed. This line is the difference, stated (M33).
 *
 * Null when the engine reported neither -- a deferral, a failure, or an older payload. An
 * absent number is not a zero.
 */
export function effort(answer: {
  attempts: number | null;
  corrected: boolean | null;
}): string | null {
  const parts: string[] = [];
  if (answer.attempts !== null) parts.push(plural(answer.attempts, "attempt"));
  if (answer.corrected !== null) {
    parts.push(answer.corrected ? "SQL corrected once" : "no repair needed");
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}
